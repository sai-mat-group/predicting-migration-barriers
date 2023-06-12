# Transfer learning project

## Outline

Investigate how well transfer learning works on small materials data sets. Is there a realtionship between the relatedness of properties and 
the effectiveness of TL? Which is more important physical relationships or data quantities? How should we do TL, on the full model (i.e. 
re-learn the representations as well as the prediction model) or just on the model head.

## Datasets

We will take our datasets from the matbenc/matminer data sources:

- gvrh
- phonons
- formation-energy
- bandgap-gga
- bandgap-experimental (expt_gap_kingsbury)
- dielectric-constant  (dielectric_constant)
- piezoelectric-modulus  [I think that we should drop this as it is just a bit too small, open to other suggestions]

For all datasets, we will keep 10% of the dataset separate and use this for testing. This data must never be used in the fine tuning stage, either 
as train or as validation data.

## Methodology

### ML training

For pre-training and fine-tuning we will use early stopping. The dataset will be split 90:10; train:validaton. The trainig is terminated if val_loss
has not improved for 50 epochs. The model with the best val_loss is used for testing.

### TL methods

We will test initally TL where the graph part of the network is frozen and the head re-trained. Later we will look at retreaining the full network.

## Experiments

### Part 1 - relations between datasets

* Cap the size of pre-training sets at 941 - the size of the smallest data set [Use 90:10 split as above]
* Pre-train a model for each of the datasets above.
* Use the pre-trained model from each dataset and do fine-tuning on each other dataset
	* Use data sizes of 10, 100, 200, 500, 800 for fine-tuning [Use 90:10 split as above]

### Part 2 - data vs physics

* Take the two largest datasets above (eform and egap), see which data they provide best and worst pre-training for
* Now increase the size of the pre-training dataset: 1000, 5000, 10000, 50000, 100000
* Compare how the performance of fine-tuning evolves for both the best/worst cases from part 1 (here use 900)

### Part 3 - re-learning representations or re-learning predictons

* Take some selected models and perform a pre-train/fine-tune (PT/FT) experiment
* Choose a pair where physics is linked (gvrh and phonons) and one where it is not (egap and phonons)
* In the FT part try relearning the full model and also just the prediction head - compare how it performs

### Part 4 - inducing structure during pre-training

* Can we use approaches to pre-structure the latent space of the network during pre-training
* Perform a PT/FT appraoch 
* Use strucutural information to bais the PT step
* PT on bandgap-gga, but with an additional loss for predicting Bravais lattice - how does this perform for gvrh, where strucutre can be important
