export OPENAI_API_KEY=cf016418d6b14bcb8165a11c97cf043e 
export OPENAI_AZURE_DEPLOYMENT=1 
python open-eqa-predict.py --method blind-gpt4 --force --verbose
python evaluate-predictions.py data/predictions/blind-gpt4-predictions.json --force --verbose
