**Autonomous Driving Kaggle Challenge Submission**
Team Members
Evangelos Vagianos
Peter Sacaleanu

**Project Description**
Over the last decade, autonomous driving has transitioned from a concept of impossibility to an inevitable future.
Following the footsteps of pioneers like Google's self-driving car project, our project aims to contribute to this evolving field by focusing on the development of a machine learning algorithm capable of autonomously navigating a test circuit. 
Utilizing a dataset of 13.8k images paired with target car responses (speed and steering angles), we trained our model to execute precise maneuvers, ensuring safe and efficient autonomous driving.

**Scenarios**
The algorithm's performance was evaluated across various scenarios, including:

Maintaining lane integrity on straight and T-junction tracks.
Halting for pedestrians on the road.
Adjusting driving behavior for pedestrians or objects off the road but nearby.
Navigating an oval track in both directions, with added complexity for pedestrian obstacles.
Executing turns at T-junctions based on traffic signs.
Driving around a figure-of-eight track without considering off-road objects.
Stopping at red traffic lights and proceeding at green.

**Evaluation Metric**
Our submissions were evaluated based on the Mean Square Error (MSE) metric, focusing on the accuracy of our predicted values, against the actual values.
