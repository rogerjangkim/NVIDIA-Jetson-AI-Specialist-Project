# NVIDIA-Jetson-AI-Specialist-Project
Project for NVIDIA Jetson AI Specialist, classification of sign language (A,B,C)

This project is based on the Thumbs Project from Nvidia's Getting Started with AI on Jetson Nano Course, we will be using the same template found from the course.
The following steps and explanations are found in the Nvidia Course:
1. Camera

This block sets the size of the images and starts the camera. If your camera is already active in this notebook or in another notebook, first shut down the kernel in the active notebook before running this code cell. Make sure that the correct camera type is selected for execution (USB). This cell may take several seconds to execute.

2. Task

You get to define your TASK and CATEGORIES (the classes) parameters here, as well as how many datasets you want to track. For the Thumbs Project, this has already been defined for you, so go ahead and execute the cell. Subdirectories for each class are created to store the example images you collect. The subdirectory names serve as the labels needed for the model. This cell should only take a few seconds to execute.

3. Data Collection

You'll collect images for your categories with your camera using an iPython widget. This cell sets up the collection mechanism to count your images and produce the user interface. The widget built here is the data_collection_widget. If you want to learn more about these powerful tools, visit the ipywidgets documentaion. This cell should only take a few seconds to execute.

4. Model

This block is where the neural network is defined. First, the GPU device is chosen with the statement:

  device = torch.device('cuda')
The model is set to the ResNet-18 model for this project. Note that the pretrained=True parameter indicates we are loading all the parameter weights for the trained Resnet-18 model, not just the neural network alone:

  model = torchvision.models.resnet18(pretrained=True)
There are a few more models listed in comments that you can try out later if you wish. For more information on available PyTorch pre-trained models, see the PyTorch documentation.

  In addition to choosing the model, the last layer of the model is modified to accept only the number of classes that we are training for. In the case of the Thumbs Project, it is only 2 (i.e. thumbs-up and thumbs-down).

  model.fc = torch.nn.Linear(512, len(dataset.categories))
This code cell may take several seconds to execute.

5. Live Execution

This code block sets up threading to run the model in the background so that you can view the live camera feed and visualize the model performance in real time. It also includes the code that defines how the outputs from the neural network are categorized. The network produces some value for each of the possible categories. The softmax function takes this vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities. The values now add up to 1 and can be interpreted as probabilities.

  output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
This cell should only take a few seconds to execute.

6. Training and Evaluation

The training code cell sets the hyper-parameters for the model training (number of epochs, batch size, learning rate, momentum) and loads the images for training or evaluation. The model determines a predicted output from the loaded input image. The difference between the predicted output and the actual label is used to calculate the "loss". If the model is in training mode, the loss is backpropagated into the network to improve the model. The widgets created by this code cell include the option for setting the number of epochs to run. One epoch is a complete cycle of all images through the trainer. This code cell may take several seconds to execute.

7. Display the Interactive Tool!

This is the last code cell. All that's left to do is pack all the widgets into one comprehensive tool and display it. This cell may take several seconds to run and should display the full tool for you to work with. This tool will look essentially the same, no matter how you set up the classification problem with this notebook.

# Our changes:
We have created a new task, sign, and defined A, B, and C as the CATEGORIES parameters for our new task.

Similarly, to collect data, we must collect images following what we would like to classify. In our case, the signs of A, B, and C in American Sign Language.
First, we would have to select the letter in which we are going to collect images for.
Once we have enough images for each letter, we will begin the training process. 
After training, we can test our model by signing the letters through the camera.
If the program is not accurately classifying the signs, we may need to collect more data and train once again.

This project is a small step towards helping people communicate with those who have trouble with their speaking by being able to spell out sign language. If a person does not know how to sign and encounters another person who has trouble speaking, this project has the potential to bridge the gap in communication. 
