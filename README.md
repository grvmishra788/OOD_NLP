# OOD_NLP

python3 main.py --energy_temp 5 --softmax_temp 10 --retrain False --debug False --output_folder "outputs"

Repo  to maintain code for CS769 course project (UW-Madison, Spring 2022)

We plan to implement a robust framework using distance-based methods to detect Out-of-distribution samples and combine/modify existing systems to increase their effectiveness. Specifically, we hypothesize that distance-based methods could be beneficial to identifying Out-of-distribution samples, and incorporating other techniques (like temperature scaling, softmax statistics analysis, etc.) in the framework would further improve the framework’s usefulness.

When NLP models are deployed in a real-world setting, they fail when the test samples do not belong to the same distribution as the ones they were trained on. For example - a Physics/Maths research article classifier, if provided with a Chemistry article, will wrongly classify the article as either Physics or Maths. The worse part is that such models often fail with high confidence predictions . So a simple probability threshold-based technique could not be used to identify such examples. Such samples are often referred to as  Out-of-Distribution (OOD) samples. The resulting unflagged, incorrect diagnoses could blockade machine learning technologies and concern AI Safety