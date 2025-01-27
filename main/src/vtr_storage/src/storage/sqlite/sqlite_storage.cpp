// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file sqlite_storage.cpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#include "vtr_storage/storage/sqlite/sqlite_storage.hpp"

#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rcpputils/filesystem_helper.hpp"

#include "vtr_storage/metadata_io.hpp"
#include "vtr_storage/serialized_bag_message.hpp"
#include "vtr_storage/storage/sqlite/sqlite_exception.hpp"
#include "vtr_storage/storage/sqlite/sqlite_statement_wrapper.hpp"

namespace {
/// \note the following code is adapted from rosbag2 foxy

// Copyright 2018, Bosch Software Innovations GmbH.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
using namespace vtr::storage;
#if false
std::string to_string(storage_interfaces::IOFlag io_flag)
{
  switch (io_flag) {
    case storage_interfaces::IOFlag::APPEND:
      return "APPEND";
    case storage_interfaces::IOFlag::READ_ONLY:
      return "READ_ONLY";
    case storage_interfaces::IOFlag::READ_WRITE:
      return "READ_WRITE";
    default:
      return "UNKNOWN";
  }
}
#endif
bool is_read_write(const storage_interfaces::IOFlag io_flag)
{
  return io_flag == storage_interfaces::IOFlag::READ_WRITE;
}

bool is_read_only(const storage_interfaces::IOFlag io_flag)
{
  return io_flag == storage_interfaces::IOFlag::READ_ONLY;
}

constexpr const auto FILE_EXTENSION = ".db3";

// Minimum size of a sqlite3 database file in bytes (84 kiB).
constexpr const uint64_t MIN_SPLIT_FILE_SIZE = 86016;

// clang-format on
}  // namespace

namespace vtr {
namespace storage {
namespace sqlite_storage {
/// \note the following code is adapted from rosbag2 foxy

// Copyright 2018, Bosch Software Innovations GmbH.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off

SqliteStorage::~SqliteStorage()
{
  if (active_transaction_) {
    commit_transaction();
  }
}

void SqliteStorage::open(
  const std::string & uri, storage_interfaces::IOFlag io_flag)
{
  if (is_read_write(io_flag)) {
    relative_path_ = uri + FILE_EXTENSION;

    // READ_WRITE requires the DB to not exist.
    if (rcpputils::fs::path(relative_path_).exists()) {
      io_flag = storage_interfaces::IOFlag::APPEND;
    }
  } else if (is_read_only(io_flag)) {  // READ_ONLY
    relative_path_ = uri;

    // READ_ONLY require the DB to exist
    // MY EDIT: also requires the file to end in .db3
    if (!(rcpputils::fs::path(relative_path_).exists() &&
      relative_path_.substr(relative_path_.length() - 4) == ".db3"))
    {
      throw std::runtime_error(
              "Failed to read from bag: File '" + relative_path_ + "' does not exist!");
    }
  }

  try {
    database_ = std::make_unique<SqliteWrapper>(relative_path_, io_flag);
  } catch (const SqliteException & e) {
    throw std::runtime_error("Failed to setup storage. Error: " + std::string(e.what()));
  }

  // initialize only for READ_WRITE since the DB is already initialized if in APPEND.
  if (is_read_write(io_flag)) {
    initialize();
  }

  // Reset the read and write statements in case the database changed.
  // These will be reinitialized lazily on the first read or write.
  read_statement_ = nullptr;
  write_statement_ = nullptr;
#if false
  ROSBAG2_STORAGE_DEFAULT_PLUGINS_LOG_INFO_STREAM(
    "Opened database '" << relative_path_ << "' for " << to_string(io_flag) << ".");
#endif
}

void SqliteStorage::activate_transaction()
{
  if (active_transaction_) {
    return;
  }

#if false
  ROSBAG2_STORAGE_DEFAULT_PLUGINS_LOG_DEBUG_STREAM("begin transaction");
#endif
  database_->prepare_statement("BEGIN TRANSACTION;")->execute_and_reset();

  active_transaction_ = true;
}

void SqliteStorage::commit_transaction()
{
  if (!active_transaction_) {
    return;
  }

#if false
  ROSBAG2_STORAGE_DEFAULT_PLUGINS_LOG_DEBUG_STREAM("commit transaction");
#endif
  database_->prepare_statement("COMMIT;")->execute_and_reset();

  active_transaction_ = false;
}

int32_t SqliteStorage::get_last_inserted_id()
{
  return database_->get_last_insert_id();
}

void SqliteStorage::write(std::shared_ptr<const SerializedBagMessage> message)
{
  if (!write_statement_) {
    prepare_for_writing();
  }
  auto topic_entry = topics_.find(message->topic_name);
  if (topic_entry == end(topics_)) {
    throw SqliteException(
            "Topic '" + message->topic_name +
            "' has not been created yet! Call 'create_topic' first.");
  }

  write_statement_->bind(message->time_stamp, topic_entry->second, message->serialized_data);
  write_statement_->execute_and_reset();
}

void SqliteStorage::write(
  const std::vector<std::shared_ptr<const SerializedBagMessage>> & messages)
{
  if (!write_statement_) {
    prepare_for_writing();
  }

  activate_transaction();

  for (auto & message : messages) {
    write(message);
  }

  commit_transaction();
}

bool SqliteStorage::has_next()
{
  if (!read_statement_) {
    prepare_for_reading();
  }

  return current_message_row_ != message_result_.end();
}

std::shared_ptr<SerializedBagMessage> SqliteStorage::read_next()
{
  if (!read_statement_) {
    prepare_for_reading();
  }

  auto bag_message = std::make_shared<SerializedBagMessage>();
  bag_message->serialized_data = std::get<0>(*current_message_row_);
  bag_message->time_stamp = std::get<1>(*current_message_row_);
  bag_message->topic_name = std::get<2>(*current_message_row_);

  ++current_message_row_;
  return bag_message;
}

bool SqliteStorage::seek_by_index(int32_t index)
{
  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.id >= " + std::to_string(index) + " "
    "ORDER BY messages.timestamp;");
  modified_message_result_ = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();
  modified_current_message_row_ = modified_message_result_.begin();
  return modified_current_message_row_ != modified_message_result_.end();
}

bool SqliteStorage::seek_by_timestamp(rcutils_time_point_value_t timestamp)
{
  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.timestamp >= " + std::to_string(timestamp) + " "
    "ORDER BY messages.timestamp;");
  modified_message_result_ = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();
  modified_current_message_row_ = modified_message_result_.begin();
  return modified_current_message_row_ != modified_message_result_.end();
}

std::shared_ptr<SerializedBagMessage> SqliteStorage::modified_read_next()
{
  std::shared_ptr<SerializedBagMessage> bag_message;
  if (modified_current_message_row_ != modified_message_result_.end()) {
    bag_message = std::make_shared<SerializedBagMessage>();
    bag_message->serialized_data = std::get<0>(*modified_current_message_row_);
    bag_message->time_stamp = std::get<1>(*modified_current_message_row_);
    bag_message->topic_name = std::get<2>(*modified_current_message_row_);
    bag_message->database_index = std::get<3>(*modified_current_message_row_);
    ++modified_current_message_row_;
  }

  return bag_message;
}

std::shared_ptr<SerializedBagMessage>
SqliteStorage::read_at_timestamp(rcutils_time_point_value_t timestamp)
{
  std::shared_ptr<SerializedBagMessage> bag_message;

  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.timestamp = " + std::to_string(timestamp) + " "
    "ORDER BY messages.timestamp;");

  auto message_result = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();

  // if(message_result.begin() == message_result.end()) {
  //   throw std::runtime_error("No messages found for timestamp " + std::to_string(timestamp));
  // }

  ModifiedReadQueryResult::Iterator current_message_row = message_result.begin();
  if (current_message_row != message_result.end()) {  // message exists
    bag_message = std::make_shared<SerializedBagMessage>();
    bag_message->serialized_data = std::get<0>(*current_message_row);
    bag_message->time_stamp = std::get<1>(*current_message_row);
    bag_message->topic_name = std::get<2>(*current_message_row);
    bag_message->database_index = std::get<3>(*current_message_row);
  }
  // if(++current_message_row != message_result.end()) {
  //   throw std::runtime_error("More than one message at the same timestamp!");
  // }

  return bag_message;
}

std::shared_ptr<SerializedBagMessage>
SqliteStorage::read_at_index(int32_t index)
{
  std::shared_ptr<SerializedBagMessage> bag_message;

  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.id = " + std::to_string(index) + " "
    "ORDER BY messages.timestamp;");

  auto message_result = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();

  // if(message_result.begin() == message_result.end()) {
  //   throw std::runtime_error("No messages found for id " + std::to_string(index));
  // }

  ModifiedReadQueryResult::Iterator current_message_row = message_result.begin();
  if (current_message_row != message_result.end()) {  // message exists
    bag_message = std::make_shared<SerializedBagMessage>();
    bag_message->serialized_data = std::get<0>(*current_message_row);
    bag_message->time_stamp = std::get<1>(*current_message_row);
    bag_message->topic_name = std::get<2>(*current_message_row);
    bag_message->database_index = std::get<3>(*current_message_row);
  }

  return bag_message;
}

std::shared_ptr<std::vector<std::shared_ptr<SerializedBagMessage>>>
SqliteStorage::read_at_timestamp_range(
  rcutils_time_point_value_t timestamp_begin,
  rcutils_time_point_value_t timestamp_end)
{
  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.timestamp BETWEEN " + std::to_string(timestamp_begin) +
    " AND " + std::to_string(timestamp_end) + " "
    "ORDER BY messages.timestamp;");

  auto message_result = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();

  auto bag_message_vector =
    std::make_shared<std::vector<std::shared_ptr<SerializedBagMessage>>>();

  ModifiedReadQueryResult::Iterator current_message_row = message_result.begin();
  for (; current_message_row != message_result.end(); ++current_message_row) {
    bag_message_vector->push_back(std::make_shared<SerializedBagMessage>());
    bag_message_vector->back()->serialized_data = std::get<0>(*current_message_row);
    bag_message_vector->back()->time_stamp = std::get<1>(*current_message_row);
    bag_message_vector->back()->topic_name = std::get<2>(*current_message_row);
    bag_message_vector->back()->database_index = std::get<3>(*current_message_row);
  }
  return bag_message_vector;
}

std::shared_ptr<std::vector<std::shared_ptr<SerializedBagMessage>>>
SqliteStorage::read_at_index_range(
  int32_t index_begin,
  int32_t index_end)
{
  auto read_statement = database_->prepare_statement(
    "SELECT data, timestamp, topics.name, messages.id "
    "FROM messages JOIN topics ON messages.topic_id = topics.id "
    "WHERE messages.id BETWEEN " + std::to_string(index_begin) +
    " AND " + std::to_string(index_end) + " "
    "ORDER BY messages.timestamp;");

  auto message_result = read_statement->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string, int32_t>();

  auto bag_message_vector =
    std::make_shared<std::vector<std::shared_ptr<SerializedBagMessage>>>();

  ModifiedReadQueryResult::Iterator current_message_row = message_result.begin();
  for (; current_message_row != message_result.end(); ++current_message_row) {
    bag_message_vector->push_back(std::make_shared<SerializedBagMessage>());
    bag_message_vector->back()->serialized_data = std::get<0>(*current_message_row);
    bag_message_vector->back()->time_stamp = std::get<1>(*current_message_row);
    bag_message_vector->back()->topic_name = std::get<2>(*current_message_row);
    bag_message_vector->back()->database_index = std::get<3>(*current_message_row);
  }
  return bag_message_vector;
}

std::vector<TopicMetadata> SqliteStorage::get_all_topics_and_types()
{
  if (all_topics_and_types_.empty()) {
    fill_topics_and_types();
  }

  return all_topics_and_types_;
}

uint64_t SqliteStorage::get_bagfile_size() const
{
  const auto bag_path = rcpputils::fs::path{get_relative_file_path()};

  return bag_path.exists() ? bag_path.file_size() : 0u;
}

void SqliteStorage::initialize()
{
  std::string create_stmt = "CREATE TABLE topics(" \
    "id INTEGER PRIMARY KEY," \
    "name TEXT NOT NULL," \
    "type TEXT NOT NULL," \
    "serialization_format TEXT NOT NULL," \
    "offered_qos_profiles TEXT NOT NULL);";
  database_->prepare_statement(create_stmt)->execute_and_reset();
  create_stmt = "CREATE TABLE messages(" \
    "id INTEGER PRIMARY KEY," \
    "topic_id INTEGER NOT NULL," \
    "timestamp INTEGER NOT NULL, " \
    "data BLOB NOT NULL);";
  database_->prepare_statement(create_stmt)->execute_and_reset();
  create_stmt = "CREATE INDEX timestamp_idx ON messages (timestamp ASC);";
  database_->prepare_statement(create_stmt)->execute_and_reset();
}

void SqliteStorage::create_topic(const TopicMetadata & topic)
{
  if (topics_.find(topic.name) == std::end(topics_)) {
    auto insert_topic =
      database_->prepare_statement(
      "INSERT INTO topics (name, type, serialization_format, offered_qos_profiles) "
      "VALUES (?, ?, ?, ?)");
    insert_topic->bind(
      topic.name, topic.type, topic.serialization_format, topic.offered_qos_profiles);
    insert_topic->execute_and_reset();
    topics_.emplace(topic.name, static_cast<int>(database_->get_last_insert_id()));
  }
}

void SqliteStorage::remove_topic(const TopicMetadata & topic)
{
  if (topics_.find(topic.name) != std::end(topics_)) {
    auto delete_topic =
      database_->prepare_statement(
      "DELETE FROM topics where name = ? and type = ? and serialization_format = ?");
    delete_topic->bind(topic.name, topic.type, topic.serialization_format);
    delete_topic->execute_and_reset();
    topics_.erase(topic.name);
  }
}

void SqliteStorage::prepare_for_writing()
{
  write_statement_ = database_->prepare_statement(
    "INSERT INTO messages (timestamp, topic_id, data) VALUES (?, ?, ?);");
}

void SqliteStorage::prepare_for_reading()
{
  if (!storage_filter_.topics.empty()) {
    // Construct string for selected topics
    std::string topic_list{""};
    for (auto & topic : storage_filter_.topics) {
      topic_list += "'" + topic + "'";
      if (&topic != &storage_filter_.topics.back()) {
        topic_list += ",";
      }
    }

    read_statement_ = database_->prepare_statement(
      "SELECT data, timestamp, topics.name "
      "FROM messages JOIN topics ON messages.topic_id = topics.id "
      "WHERE topics.name IN (" + topic_list + ")"
      "ORDER BY messages.timestamp;");
  } else {
    read_statement_ = database_->prepare_statement(
      "SELECT data, timestamp, topics.name "
      "FROM messages JOIN topics ON messages.topic_id = topics.id "
      "ORDER BY messages.timestamp;");
  }
  message_result_ = read_statement_->execute_query<
    std::shared_ptr<rcutils_uint8_array_t>, rcutils_time_point_value_t, std::string>();
  current_message_row_ = message_result_.begin();
}

void SqliteStorage::fill_topics_and_types()
{
  auto statement = database_->prepare_statement(
    "SELECT name, type, serialization_format FROM topics ORDER BY id;");
  auto query_results = statement->execute_query<std::string, std::string, std::string>();

  for (auto result : query_results) {
    all_topics_and_types_.push_back(
      {std::get<0>(result), std::get<1>(result), std::get<2>(result), ""});
    topics_.insert(std::make_pair(std::get<0>(result), 1));
  }
}

std::string SqliteStorage::get_storage_identifier() const
{
  return "sqlite3";
}

std::string SqliteStorage::get_relative_file_path() const
{
  return relative_path_;
}

uint64_t SqliteStorage::get_minimum_split_file_size() const
{
  return MIN_SPLIT_FILE_SIZE;
}

BagMetadata SqliteStorage::get_metadata()
{
  BagMetadata metadata;
  metadata.storage_identifier = get_storage_identifier();
  metadata.relative_file_paths = {get_relative_file_path()};

  metadata.message_count = 0;
  metadata.topics_with_message_count = {};

  auto statement = database_->prepare_statement(
    "SELECT name, type, serialization_format, COUNT(messages.id), MIN(messages.timestamp), "
    "MAX(messages.timestamp) "
    "FROM messages JOIN topics on topics.id = messages.topic_id "
    "GROUP BY topics.name;");
  auto query_results = statement->execute_query<
    std::string, std::string, std::string, int, rcutils_time_point_value_t,
    rcutils_time_point_value_t>();

  rcutils_time_point_value_t min_time = INT64_MAX;
  rcutils_time_point_value_t max_time = 0;
  for (auto result : query_results) {
    metadata.topics_with_message_count.push_back(
      {
        {std::get<0>(result), std::get<1>(result), std::get<2>(result), ""},
        static_cast<size_t>(std::get<3>(result))
      });

    metadata.message_count += std::get<3>(result);
    min_time = std::get<4>(result) < min_time ? std::get<4>(result) : min_time;
    max_time = std::get<5>(result) > max_time ? std::get<5>(result) : max_time;
  }

  if (metadata.message_count == 0) {
    min_time = 0;
    max_time = 0;
  }

  metadata.starting_time =
    std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds(min_time));
  metadata.duration = std::chrono::nanoseconds(max_time) - std::chrono::nanoseconds(min_time);
  metadata.bag_size = get_bagfile_size();

  return metadata;
}

void SqliteStorage::set_filter(
  const StorageFilter & storage_filter)
{
  storage_filter_ = storage_filter;
}

void SqliteStorage::reset_filter()
{
  storage_filter_ = StorageFilter();
}

// clang-format on

}  // namespace sqlite_storage
}  // namespace storage
}  // namespace vtr