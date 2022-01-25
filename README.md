# phase-1-hnscc-trial-prediction
Create and evaluate exploratory prediction model for phase 1 HNSCC clinical trial.

The code in this repository produces the results of the random forest model in the paper titled: "Towards a mechanistic understanding of patient response to neoadjuvant SBRT with anti-PDL1 in human HPV-unrelated locally advanced HNSCC: phase I/Ib trial results"

To run the code perform the follow:
1) Clone the repository
2) Download the data (at the time of writing it is TBD how this will be done)
3) Change directory to the repository
4) Install the requirements `pip install -r requirements.txt`
5) Run the script with `python3 generate_results.py`
    5a) Optionally, you may edit arguments of where to write files or the number of trials to run by editing the variables at the top of the script.
