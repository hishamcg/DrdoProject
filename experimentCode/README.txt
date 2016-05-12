You are encouraged to use our data set for your own feature exctraction and classifiers, but if you want some baseline to compare your results, or at least a starting point, you may follow the steps below to replicate our process for training models and testing on new data.

To run these programs, you may have to do some source code editing to make it work on your particular machine.  If you have your own feature extraction mechanisms, they are recommended.

To create the process for which we trained our model and extracted our features, follow these instructions:


1) Extract your features from your training data.  In order to do this, you need to separate your training data from your test data, numbered sequentially starting with 0000.  Then you extract features from the labelled grasping rectangles either using your own code, or Marcus Lim's.  To use his features, open up your shell with ROS and OpenCV installed, and navigate to the svmlight directory (in the LimClassifier folder.  svmlight is the name of this ROS Package).  To extract features, call rosrun svmlight extract_pc_features folder numFiles.  Where:
	folder is the location of the raw data files (images, point clouds, and labelled rectangles both positive and negative).  There must NOT be anything else in this folder or this will not work as intented.
	numFiles is the number of data files.  Your first image must be pcd0000r.png.  Your first point cloud must be pcd0000.txt.  Your first positive and negative rectangle files must be pcd0000cpos.txt and pcd0000cneg.txt respectively.  Then they must increment by 1: 0001, 0002, 0003, etc...  If the last one is 0400, then numFiles is 401, since it goes from 0 to 400.
example:
rosrun svmlight extract_pc_features ~/data 1000


The output from extract_pc_features is a single file with the rewards and features for each rectangle.

2)  Train your model.  Use SVM-Light for this.  We have included the executable for this.  To run with the features you extracted in (1), open up your shell and navigate to the svm_light directory (not the rospackage containing Lim's code!).  This directory should have two programs: svm_classify and svm_learn.  You can get instructions on how to run svm_learn on the author's webpage at svmlight.joachims.org.  The output of svm_learn is a model file.  This model file is what you need to rank rectangles in new images.

3)  Search for grasping rectangles given a new image and point cloud.  Read the README.txt in the LimClassifer/svmlight directory for instructions on how to use the rank program.  Essentially it will look like: rank imgFile bgimgFile pointcloud svmtrainedmodel saveDirectory

Rank will save a file with all of the top 100 grasping rectangles and their scores using the same feature set as extract_pc_features from step 1.  It will also save 2 images in the specified directory: an image with the top 10 rectangles superimposed over top of it, and an image with just the top rectangle superimposed.

Feel free to edit the source files and remake the package with your own changes.  You may want to discretize the space less and exhaustively search more rectangles, or you may want to add/change the features. Or you may want to disable background subtraction.  Instructions on building a ROS package can be found at ros.org.  

We have included SVM-Light to work as our classifier.  It is by:
       T. Joachims, Making large-Scale SVM Learning
       Practical. Advances in Kernel Methods - Support Vector
       Learning, B. Schölkopf and C. Burges and A. Smola (ed.),
       MIT-Press, 1999. 
       http://www-ai.cs.uni-dortmund.de/DOKUMENTE/joachims_99a.pdf
