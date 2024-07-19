export OPENAI_API_KEY=<YOUR_API_KEY> 
export OPENAI_AZURE_DEPLOYMENT=1 
python open-eqa-predict.py --method gpt4o --force --verbose
python evaluate-predictions.py data/predictions/gpt4o-predictions.json --force --verbose
