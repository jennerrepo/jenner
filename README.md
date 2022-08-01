## Jenner README

### Setup
Install PostgreSQL v12+

Installing the Python libraries
```
pip install numpy, scipy
pip install scikit-learn
```

Extract classifiers and data

```
cd codebase
unzip Classifiers.zip
unzip Data.zip
```

### About the code 

- The code consists of multiple trained classifiers as the enrichment functions (added to the “Classifiers”directory), 
  and several Python scripts containing the source code(added to the base directory).
- Inside the “Classifiers” directory, we have added two sub-directories of “Image” and “Tweet” each containing the enrichment 
  functions for the Image and Tweet dataset respectively. 
- The dataset is stored as a pickle file (compressed manner) in the “Data” directory.

### Running the code

In order to execute Q4-Q7 of the paper which were on Tweet dataset, please run the  python script named `QueryExecutionTweet.py`.
```
python QueryExecutionTweet.py 
```
Note: by default this code uses TweetDataSmall.p file which is a 4 MB file. The bigger dataset of TweetDataBig.p can also be used with this script. The data used in the experiments had the size of 10.5 GB. We could not upload the complete dataset in this repo due github space limitations. To get the bigger dataset contact us at dhruajg@uci.edu or peeyushg@uci.edu. 


In order to execute Q8-Q12 of the paper which were on Image dataset, please run the  python script neamed `QueryExecutionMultiPie.py`.
```
python QueryExecutionMultiPie.py 
```

