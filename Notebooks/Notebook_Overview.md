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
This notebook is desighned to look at the across run rest fMRI SWC data and find out why clustering based on runs happens when using LE. The notebook first looks at the original 200 ROI time series of the concatinated data as a carpet plot and the SWC matrix to see if anything stands out. When looking at these plots you can't see any lines of data that occure for only one run and not the others.
We then tested to see that if we added surogate data with elevations in signal over certain time periods, would LE be able to detect these fake classifiers. When we add around 0.4% new data to the origninal SWC data, the embedding takes on clusters based on these "fake" classifier. This told us that you only need a small fraction of data points that have destinct signal elevation for LE to pick it up as clusters in the data.
To try and identify the connections that had elevated signals over certain runs, we created a test connection that had an elevated signal of 1 for a run period and 0 for the rest of the connection, and correlated that will all the other connections. We expected that connections that were highly corrlated or highly anti-correlated with the test connections could be the connection that containted an elevated signal and was causing the clustering based on runs. For the the subjects we looked at (24, 26, and 7) the highest correlation computed was around 0.5 and the highest anti-correlation was -0.5. We removed these connections, with high correlaiton and anti-correlation, and computed the embedding using LE. To finally see run specific cluster be removed from the data we had to remove aaround 800 connections for each run (6 runs) out of aproximatley 9000 total connections. We were usally left with around 1/2 the original connections. We also checked to see if removing the same amount of connections also caused the run specific clustering to be non exixtant in the embedding but it was not. This tells us that the connections we removed did in fact posses run specific information. Although this is helpful we were suprised to see so many connections removed.
We did this again except with SWC data computed without PCA. We saw similar findings excpet we had to remove around 2000 connections per run. Although these are more connections, the data has more connections wince it was computed without PCA.
We allso did this for the raw 200 ROI time series data but the correlation vlaues were so small it was not helpful at destiguishing ROI's that had elevated signal for a given run.

***
## Run Effect on Simulated Data (*SE_Simulation_NConns.ipynb*)
This noteboook was created by Javier and is used to observe the run effect when computing embeddings usising LE on simulated data. The finding was that even a very small percent elevated run data is picked up by LE. Javier posses the updated version of this notebook with more options.

***
## t_SNE on All Data Sets (*t-SNE_Scikit.ipynb*)
This function is designed to compare the t-SNE embedding for all four data sets (digits, fashion, rest fMRI, task fMRI). All t-SNE are computed the same for each data set using default Scikit Learn peramiters (Euclidean distance metric) and the user is able to choose what data set they wish to see plotted, the perplexity value, and the learning rate. For some data sets, such as the rest and task fMRI, the user is also able to choose options for that specific data type. Things like subject, run, and window length.
Findings:
* The perplexity value behaves very similarly to the k value in LE. Small perplexity leads to speghetti like embedding cuased by auto correlation. When perplexity is increased one is able to detect better clustering.
* t-SNE worked best for the task based data. Clustering based on task was very clear with perplexity around 70. Auto correlaiton was still present but within task clusters. This is unlike LE. Pure task windows showed the clusters clearer.
* Similar to LE there was no clustering based on sleep stage in the rest data set.
* I have not been able to pin point the exact effects of changing the learning rate. It seems to do little to the data when changes. However too big of a learning rate causes messy data.

***
## Activity Based Multi-Task fMRI Data (*Task_fMRI_activity.ipynb*)
The notebook starts by computing the activity data for each subject. Over a given window (30 or 46 seconds) the activity for each ROI is computed as the SD of the signal in that window of time. The activity data is saved to '/data/SFIMJGC/PRJ_Manifold_Learning/Data/MultiTask/', one file per subject. We then compute the LE and t-SNE embedding for the acctivity data with default setting for each technique. Widgets for the hyper peramiter for each technique (k, perplexity, and learning rate) allow the user to choose these peramiters.
Findings:
* Clustering of task is aparent in the data however it is less clear.
* Better clustering might be seen if a differnt method of computing activity is used.

***
## Computing Intrisic Dimension of the Data (*Intrinsic-Dimension.ipynb*)
Using the sciki-dimemsion package we compute the ID for both the the task and rest SWC data with PCA. The user must select the subject, run (for rest only), window length in seconds, and window purity (for task only) to load the SWC data. Then the scikit-dim function is performed on the data sets. A k value must be deffined to compte the LE. The dimension is then printed for both data sets.