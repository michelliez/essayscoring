# Methods used to improve LB Scores
The methods are implemented in the notebooks in this folder.

## Thresholding
Used code from Kaggle notebook (https://www.kaggle.com/code/roschildrui/deberta-only-thresholds-optimize-lb-0-81) to find best thresholds for QWK score optimization. Improved QWK on baseline DeBERTa v3 small model from 0.7950 to 0.80092.

## Pooling
- First tried naive average of last 5 hidden layers, then to Mean + CLS Pooling. Concatenated mean pooling and cls pooling for logits. RESULT: For fold 0, epochs performed worse than the base model. Reverted back to mean pooling.
- Tried just Mean + CLS Pooling

## Error Analysis
- Found the absolute error between predictions and labels for the validation set
- Created a DataFrame and sorted by error
- Took the last 30 (the 30 essays that were scored poorly) and printed those essays
**EXAMPLE ESSAY**
Index: 1418 
Score: 2.0 
Prediction: 3.6876487731933594 
Essay: The media is everywhere. Everyday, you hear gossip about celebrities in magazines, movies, and social media. Millions of movies and magazines are released all of the time around the world. More than half of the time, the facts they tell us aren't even close to accurate. You can't always trust the Media, especially when including scientific facts. Twenty five years ago something peculiar happened on Mars. NASA's Viking 1 spacecraft was circling around Mars, taking pictures, and spotted a Martian Mesa. The Media is lying to all of us. Are they honestly trying to get us to believe that the Mesa is something that was created by aliens? That's purposturous. If "aliens" did however create the Mesa, NASA would benifit from it.

You can't trust information from movies or magazines because most of the time, they are false accuzations. Most of the "facts" they tell us about are completely untrue. There isn't enough scientificly proven facts to believe what magazine and movie writers are telling us. The Mesa is just a huge rock formation, not an alien created object. NASA is a very sophisticated scientific group, trust the information they tell us, not false information.

I understand that few scientists believe that aliens somehow created the Mesa. But, an extremely large amount of scientists highly disagree. There is absolutley no scientific proof of aliens created the Mesa on Mars. NASA has stated that it's just a rock formation. if aliens created it, NASA would release that information.

On April 5, 1998, Michael Malin took his Mars Orbiter Camera to fly to Cydonia where he would take a picture of the rock formation. He later revealed on a JPL website, that it was just a nautral landform. NASA's information is always correct. Thanks to Michael Malin, we have proof that the Mesa is just a rock formation.

In conclusion, NASA has provided all of us with many accurate facts about the Martian Mesa. Some people still choose to argue, even though NASA has already provided us with all of the right answers. I hope I have pursuaded you to think more into the facts that are scientifically correct, unlike the Media's version of the story.   
look at the rows in the validation set that scored poorly and see (like read those essays) and see if there's some pattern that you can pick up that can help you get a better score
It would be more or less asking whether there's a persistent failure mode - e.g. your model tends to predict 2 rather than 3, or long essays have a tendency to score worse than short ones that fit in the context length, etc.
- This essay has several spelling errors. Further, the sentences don't make sense. They don't add onto the previous sentence's ideas.

## Ensemble of different seeds
- Averaged predictions from seed 42 and seed 23. Then used thresholding. CV score was much higher than Private LB score. CV: 0.85; Private: 0.8016. I realized this is because different seeds yield different splits and cause leakage. Regular average will not cause leakage, but this method of thresholding onto mismatched folds caused leakage and a higher than expected LB score.