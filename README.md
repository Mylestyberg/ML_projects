# ML_projects
 Project Cars (Racing Game) - Using DQN with CNN 
To do:
- mss to increase fps

Resources/Help 
Big thanks to all these resources for helping me build the project so far:
- Heavily relied on this medium series to build my first project and learn the core concepts of ML https://medium.com/@carsten.friedrich/teaching-a-computer-to-play-tic-tac-toe-88feb838b5e3
- Use carsouer API to get info from game - Thank you https://github.com/matslindh/carseour
- https://lopespm.github.io/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html - This helped me find a reward function for the car.
- Sentdex, https://pythonprogramming.net, the tutorials have helped me learn the techniques needed for ML, especially the GTA V self driving car and reinforcement learning series.




Log - Issues:
- A prior project to this was using reinforcement learning to teach an agent to play tic tac toe, I used the same structure for the environment I use there, here.
- I tried using Pytesseract and Google OCR to detect text from image so I could get speed and percentage of track completed, this was going to be how I originally was going to set up the reward functions. Unfortunately both these problems suffered from issues. Pytesseract was incorrect due to the noise coming out of the image, in order to try to remove this I worked through using opencv to try and solve this. Unfortunately the text detection still wasnt good enough. I then use google OCR, this worked amazingly, but the detection time took 1 second and caused the program to lag. I ended up finding an api which dealt with problems of receiving information from the game , Carsouer, this api allowed me to see when the car crashed and the speed of car. I changed the reward function based off an article I found which incorporated both these outputs of the car. (both article and Carsouer can be found in resources).

