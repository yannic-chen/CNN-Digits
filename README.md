# CNN Digits handwriting classification

This project is an extension of the [4th lesson](https://course.fast.ai/videos/?lesson=4) from [FastAI 2020 course](https://course.fast.ai/). Here the mnist dataset was used to train a CNN to classify handwritten digits. 

The trained model is exported out ("export.pkl") and loaded python script which creates a GUI tool (Tkinter). This tool allows the user to handwrite a digit onto the given canvas and then let the computer predict the number based on the trained model. THe resulting "guess" is shows with a probability (as well as the probability of all digits).

*Care must be taken when drawing as to make the digit large and centered. Off center numbers are difficult for the computer to recognize.*
