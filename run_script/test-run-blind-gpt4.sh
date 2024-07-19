export OPENAI_API_KEY=<YOUR_API_KEY> 
export OPENAI_AZURE_DEPLOYMENT=1 
python open-eqa-predict.py --method blind-gpt4 --force --verbose --dry-run
python evaluate-predictions.py data/predictions/blind-gpt4-dry_run-predictions.json --force --verbose --dry-run
