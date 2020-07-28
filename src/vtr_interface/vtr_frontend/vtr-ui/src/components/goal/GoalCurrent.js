import React from "react";
import { withStyles } from "@material-ui/core/styles";
import Card from "@material-ui/core/Card";
import CardActions from "@material-ui/core/CardActions";
import CardContent from "@material-ui/core/CardContent";
import Button from "@material-ui/core/Button";
import Typography from "@material-ui/core/Typography";
import clsx from "clsx";

const styles = (theme) => ({
  root: {
    minWidth: 275,
  },
  bullet: {
    display: "inline-block",
    margin: "0 2px",
    transform: "scale(0.8)",
  },
  title: {
    fontSize: 14,
  },
  pos: {
    marginBottom: 12,
  },
});

class GoalCurrent extends React.Component {
  render() {
    const { classes, className } = this.props;
    return (
      <Card className={clsx(classes.root, className)}>
        <CardContent>
          <Typography variant="h5" component="h2">
            {Object.keys(this.props.currGoal).length === 0
              ? "No Goal"
              : "Has a goal"}
          </Typography>
        </CardContent>
        <CardActions>
          <Button size="small" onClick={(e) => this.props.setCurrGoal({}, e)}>
            Cancel
          </Button>
        </CardActions>
      </Card>
    );
  }
}

export default withStyles(styles)(GoalCurrent);
