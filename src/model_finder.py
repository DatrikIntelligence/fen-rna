import os
import sys
import json
import argparse
import logging
import time

logging.basicConfig(level = logging.INFO)
logging.info("Working dir: " + os.getcwd())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Adding optional argument
    parser.add_argument("-m", "--model", help = "Model params", type=str, required=True)
    parser.add_argument("-d", "--dataset", help = "Dataset params", type=str, required=True)
    parser.add_argument("-g", "--gpu", help = "If use gpu", action='store_true')
    parser.add_argument("-c", "--cuda", help = "Cuda visible", choices=["0", "1"], default="", required=False)
    parser.add_argument("-nc", "--ncpus", help = "CPUs to take", type=int, required=False, default=4)
    parser.add_argument("-ng", "--ngpus", help = "GPUs to take", type=int, required=False, default=2)
    
    # Read arguments from command line
    args = parser.parse_args()
    
    model_config = json.load(open(args.model, 'r'))
    dataset_config = json.load(open(args.dataset, 'r'))
    logging.info("Params read")
    
    logging.info("GPU: " + str(args.gpu))
    if args.cuda != "":
        os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
        
        
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    
    import utils
    import models
    def train(config):
        net_creator_func = f"create_{model_config['net']}_model"
        func = getattr(models, net_creator_func)
        
        return utils.parameter_opt_cv(func, config, dataset_config)
    
    space = {k: (i,e) for k, (i,e) in model_config['opt_params'].items()}
    points_to_evaluate = model_config['points_to_evaluate']
    optimization_config = {
                "working_dir": os.getcwd(),
    }
    optimization_config.update(model_config['fixed_params'])
    
    from ray import tune
    from ray.tune.logger import CSVLoggerCallback
    import ray
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.schedulers import ASHAScheduler
    import models ##import create_mscnn_model
    

    ray.init(num_cpus=args.ncpus, num_gpus=args.ngpus)
    
    bayesopt = BayesOptSearch(space=space, mode="min", metric="score", 
                              points_to_evaluate=points_to_evaluate,
                              random_search_steps=20)

    csvlogger = CSVLoggerCallback()
    
    def trial_str_creator(trial):
        trialname = model_config['net'] + "_" + dataset_config['name'] + "_" + trial.trial_id
        return trialname
    
    analysis = tune.run(
        train,
        name=model_config['net'] + "_" + dataset_config['name'],
        config=optimization_config,
        resources_per_trial={'gpu': 1 if args.ngpus > 0 else 0, 'cpu': min(2, args.ncpus)},
        num_samples=100,
        search_alg=bayesopt,
        callbacks=[csvlogger],
        log_to_file=False,
        trial_name_creator=trial_str_creator,
        storage_path=os.path.join(os.getcwd(), "..", "results/tune/opt"),
    )
    
