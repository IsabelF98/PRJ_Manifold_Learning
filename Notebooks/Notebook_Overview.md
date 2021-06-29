# Notebook Descriptions and Findings
Isabel Fernandez 6/29/2021

***
## Observing Data Sets (*Comparing_Data.ipynb*)
The purpose of this notebook is see how the distributions of the correlations and distances of different data sets work. We observe 4 differnet data sets, digits, fashion, Samika's resting fMRI (3T), and mulit task fMRI (7T).
Findings:
* Digits: Distance and correlation matrix are very similar. Correlation distribution is fairly gausian, centered around 0.5. Slight bump at higher correlation. Distance matrix is skewed to the right gausian, centered around 50.
* Fashion: Distance and correlation matrix show similar features. Correlation distribution has two bumps centered around 0.4. Distance distribution is gausian, centered around 2700.
* Rest fMRI: Strong auto correlation is aparent in both distance and correlation matrix. Decreasing the range of the correlation values to -0.2 to 0.2 help destiguish other apparent correlations such as correlation by run. Distribution of correlaitons is very skewed to the left around -0.1 with a long and small positive tail. Distance distribution is very skewed to the right centered around 32.
* Multi Task fMRI: Strong auto correlation is aparent in both distance and correlation matrix. Decreasing range of correlation values to to -0.2 to 0.2 help destiguish tasked based correlation. When using pure windows the correlations between like tasks are easier to see. Both distributions are similar to rest fMRI distributions.

***
## Laplacian Eigenmap on Digits Data Set (*Laplacian_Eigenmap_Scikit_1.ipynb*)
This was one of the first notebooks created to understand LE. We looked at many differnt sets within the digits data set such as the whole set, only the 1, 3, 7, and 8 digits, and a smaller number of data points then original dimensions. The first part of the notbook creates a GUI where one is able to choose a differnt eigenvector solver function (options in scikit learn) and k value for computing k nearest neighbor. The GUI plots two LE embeddings, one using the default Euclidean distance metric and the other using the correlation distance metric (these affinity matricies had to be precomputed since they take so much time to run). The acompanying affinity matrix for the two displayed embeddinngs are also plotted on the GUI.
Finidings:
* The two differnt distance matrics do play that much a difference in the the embedding outputs. For alot of them they are the same or similar. For some the correlation metric might work a little better but its hard to tell.
* A k value that is too small (around 3) will cause a data points to be left disconected.
* A k vlaue that is too big (more then the number of samples for the smallest number of a certain digit) causes data points to cluster with other digit. This makes it harder to destinguish clusters.
We also used the notebook to understand the affinity matrix better. The affinitny matrix is an important part of computing LE since it is what is used to compute the graph Laplacian. The affinity matrix must be semetric for it to be used. In some cases a data point that is found to be the neighbor of another data point might not be not have the other data point in the set of its neighbors. Scikit Learn makes the affinity matrix symetric by taking the average of the affinity matrix by its transpose. This makes every non symmetric connection bweteen point 0.5 which play no effect in the computing of the LE. We also tried using correlation matrix and then applying an eplison criteria to create an affinity matrix. This methon also worked will in creating an embedding.

***
## Laplacian Eigenmap on All Data Sets (*Laplacian_Eigenmap_Scikit_2.ipynb*)
This function is designed to compare the LE embedding for all four data sets (digits, fashion, rest fMRI, task fMRI). All LE are computed the same for each data set using default Scikit Learn peramiters (Euclidean distance metric) and the user is able to choose what data set they wish to see plotted, eigenvector solver function (options in scikit learn) and k value for computing k nearest neighbor. For some data sets, such as the rest and task fMRI, the user is also able to choose options for that specific data type. Things like subject, run, window length.
Findings:
* The digits data set and task data set cluster the best.
* The rest data set clusters based on run (something we did not expect to see).
* The fashion data set does cluster but its hard to destiguish clusters without coloring. This could be because there are 10 groups to cluster by.
The last time I treid running this notbook I coulg not get the GUI to display.

***
## Load Fashion Data Set (*Load_Fashion_Data.ipynb*)
This notbook loads the fashion data set from tenserflow and saves it to 'PRJ_Manifold_Learning/Data/Fashion_Data/'. Both the test and training data set are loaded.

***
## Looking at Rest fMRI data (*Rest_fMRI_plots.ipynb*)
