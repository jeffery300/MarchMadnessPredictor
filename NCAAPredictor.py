from __future__ import with_statement
from __future__ import division
from collections import defaultdict
import csv

def learn_naive_bayes(class_key,feature_keys,
                      dataset,
                      class_prior_count=1,feature_posterior_count=1,
                      class_domain=None,feature_domains=None):
    if class_domain == None:
        #compute the set of values that the class can take on
        class_domain = set([instance[class_key] for instance in dataset])
    if feature_domains == None:
        #compute the set of values that the features can take on
        feature_domains = dict()
    for f in feature_keys:
        if f not in feature_domains:
            feature_domains[f] = set([instance[f] for instance in dataset])
    


    #learns the class prior
    PClearned = learn_discrete([instance[class_key] for instance in dataset],class_prior_count,class_domain)
    PFlearned = dict()
    
    featurelist = []



    for f in feature_keys:
        tempdict = {}
        for clss in class_domain:
            tempdict[clss] = learn_discrete([instance[f] for instance in dataset if instance[class_key] == clss],feature_posterior_count,feature_domains[f])
        PFlearned[f] = tempdict

    return (PClearned,PFlearned)

def learn_discrete(dataset,virtual_count=1,domain=None):
    """Given a list of values in dataset, learns a discrete distribution
    over the value."""

    if(domain==None):
        domain = set(dataset)

    discreteDstr = {}
    denom = len(dataset) + (virtual_count*len(domain)) #add virtual count once 
    for label in domain:
        count = 0
        for l in dataset:
            if(l == label):
                count = count +1
        discreteDstr[label] = (virtual_count + count)/denom

    return discreteDstr

def naive_bayes(class_probabilities,feature_probabilities,instance):
    """Naive Bayes inference. Given class probabilities P(C) and feature
    conditional probabilities P(Fk|C), compute P(C|F=finstance), where
    - P(C=c) = class_probabilities[c]
    - P(Fk=fk|C=c) = feature_probabilities[feature_name][c][feature_value]
      with feature_name=Fk, feature_value=fk, for all k=1,...,n
    - finstance = instance
    and all features F1,...,Fn are assumed independent, given C.
    """
    #TODO: compute P(C|F=f)
    #returns one proabilibity value
    answer = {}
    for clss,prob in class_probabilities.iteritems():
        answer[clss] = prob
    
    for clss,clssprob in class_probabilities.iteritems():
        for feature,true in instance.iteritems():
            answer[clss] = answer[clss]*feature_probabilities[feature][clss][true]
            
    answer = normalize(answer)
    return answer

def normalize(probabilities):
    """Given an unnormalized distribution, returns a normalized copy that
    sums to 1."""
    vtotal = sum(probabilities.values())
    return dict((k,v/vtotal) for k,v in probabilities.iteritems())


def loadgames(fn="2017 NCAAM Game Results Data.csv"):
    #load stats
    games = []
    with open(fn,"r") as csvfile:
        games = [row for row in csv.DictReader(csvfile)]
    return games

def loadfeatures(fn="2017 NCAAM Game Results Features.csv"):
    #load features
    games = []
    with open(fn,"r") as csvfile:
        games = [row for row in csv.DictReader(csvfile)]
    return games


namevars = ["Date","Team","Opponent"]

#Roots of stat variables.  These will appear in the games file as keys
#ScoreOff, ScoreDef, RushAttOff, RushAttDef, ...
#and in the features files as team_avg_Score, team_avg_ScoreAllowed, opp_avg_Score, ...,
#
statvars = ["Score"]
statvars_off_def = ["Team "+s for s in statvars] + ["Opponent "+s for s in statvars]
statvars_team_avgs = ["team_avg_"+s for s in statvars] + ["team_avg_"+s+"Allowed" for s in statvars]
statvars_opp_avgs = ["opp_avg_"+s for s in statvars] + ["opp_avg_"+s+"Allowed" for s in statvars]


def betterStatThanOpponent(gamefeatures,item):
    """Returns True if the given stat is better than the opponent's stat"""
    return float(gamefeatures['team_avg_'+item]) > float(gamefeatures['opp_avg_'+item])

def transformToBooleanFeatures(gamefeatures):
    """Transforms the game features to boolean features regarding whether you are doing better or worse than the opponent"""
    res = dict()
    record_vars = ['team_wins','team_losses','opp_wins','opp_losses','Team Differential','Opponent Differential']
    for (f,v) in gamefeatures.iteritems():
        if f not in statvars_team_avgs and f not in statvars_opp_avgs and f not in namevars and f not in record_vars:
            res[f] = int(v)
    res["Differential_better"] = 1 if gamefeatures["Team Differential"] > gamefeatures['Opponent Differential'] else 0
    for f in statvars:
        res[f+"_avg_better"] = 1 if betterStatThanOpponent(gamefeatures,f) else 0
        res[f+"Allowed_avg_better"] = 1 if betterStatThanOpponent(gamefeatures,f+"Allowed") else 0
    res["wins_better"] = 1 if int(gamefeatures['team_wins']) >= int(gamefeatures['opp_wins']) else 0
    res["losses_better"] = 1 if int(gamefeatures['team_losses']) > int(gamefeatures['opp_losses']) else 0
    return res

origfeatures = loadfeatures()
transformedfeatures = [transformToBooleanFeatures(f) for f in origfeatures]

#the prediction variables should be taken from this set
non_name_variables = [f for f in transformedfeatures[0].keys() if f != "team_won"]


def classifier_accuracy(probabilistic_classifier,p_threshold,testset,target):
    """Given a probabilistic classification function such that the target
    concept of an instance x is predicted to be positive if f(x)>p_threshold,
    compute accuracy statistics on the given test set.

    testset is a list of instances, where each instance is a dictionary with
    the boolean concept given by the key specified in target.

    Return value is a dictionary with elements:
    - tp: # of true positives
    - fp: # false positives
    - tn: # true negatives
    - fn: # false negatives
    - precision: precision
    - recall: recall
    - accuracy: overall accuracy
    """
    n = len(testset)
    num_tp = 0
    num_fp = 0
    num_tn = 0
    num_fn = 0
    for item in testset:
        #this is the predicted label of the item
        predict_pos = (probabilistic_classifier(item) > p_threshold)
        #this is the actual label of the item
        actually_pos = item[target]
        #TODO: fill me in

        if predict_pos == True and actually_pos == True:
            num_tp = num_tp+1
        if predict_pos == False and actually_pos == False:
            num_tn = num_tn+1
        if predict_pos == False and actually_pos == True:
            num_fn = num_fn+1
        if predict_pos == True and actually_pos == False:
            num_fp = num_fp+1


    if (num_tp+num_fp)>0:
        precision = (num_tp)/(num_tp+num_fp)
    else:
        precision = 0
    recall = num_tp/(num_tp+num_fn)
    accuracy = (num_tp+num_tn)/(n)
    return {'tp':num_tp,'fp':num_fp,
            'tn':num_tn,'fn':num_fn,
            'precision':precision,'recall':recall,'accuracy':accuracy}

def learn(prediction_variables,virtual_counts=1,p_threshold=0.5,print_result=True):
    """Do the learning on the given prediction variables.
    Returns the Naive Bayes parameters and the training accuracy."""
    global transformedfeatures
    print "Learning on",prediction_variables

    #Do the learning
    (pWon,pFeatures)=learn_naive_bayes("team_won",prediction_variables,
                                  transformedfeatures,
                                  virtual_counts,virtual_counts)


    #Print the probability distributions
    if print_result:
        print "Prior of winning:",pWon[1]
        for (f,PF) in pFeatures.iteritems():
            print "Posterior of",f,"given win:",PF[1][1]
            print "            "," "*len(f),"given loss:",PF[0][1]


    #compute the accuracy
    def eval_probability(x):
        justfeatures = dict((f,x[f]) for f in pFeatures.keys())
        return naive_bayes(pWon,pFeatures,justfeatures)[1]

    stats = classifier_accuracy(eval_probability,p_threshold,transformedfeatures,"team_won")
    if print_result:
        n = len(transformedfeatures)
        print
        print "Training error:"
        print "%d/%d true positives, %d/%d true negatives"%(stats['tp'],n,stats['tn'],n)
        print "%d/%d false positives, %d/%d false negatives"%(stats['fp'],n,stats['fn'],n)
        print "Precision: %f, recall: %f"%(stats['precision'],stats['recall'])
        print "Total accuracy:",stats['accuracy']
    return (pWon,pFeatures,stats['accuracy'])

if __name__=="__main__":
    # ['losses_better', 'wins_better', 'at_home', 'at_opp', 'Differential_better', 'Score_avg_better', 'ScoreAllowed_avg_better']
    #TODO: play around with which variables to include in prediction
    #this line uses all variables
    prediction_variables = non_name_variables
    #prediction_variables = ['Differential_better','wins_better','at_home','Score_avg_better']
    #prediction_variables = []

    learn(prediction_variables,p_threshold=0.5)

