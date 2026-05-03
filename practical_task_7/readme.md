cat > practical_task_7/README.md << 'EOF'
# Practical Task 7: Batch Prediction Pipeline

A complete batch prediction pipeline that processes iris flowers automatically on a schedule.

## What it does

1. **Database** — SQLite stores input flowers and prediction results
2. **Batch processing** — reads unprocessed flowers from `input_data` table
3. **Model prediction** — loads trained model and predicts all at once
4. **Results storage** — saves predictions + timestamps to `predictions` table
5. **Automatic scheduling** — runs every 5 minutes via Python scheduler

## Files

- `practical_task_7.ipynb` — complete notebook with all sections
- `model.joblib` — trained Logistic Regression model (reused from Practice 6)

## How to run

```bash
jupyter notebook practical_task_7.ipynb
```

Then run all cells. The pipeline will:
- Create SQLite database
- Insert 30 sample iris flowers
- Run batch prediction
- Simulate new data arriving
- Show scheduling working (5-minute intervals)

## Database schema

### input_data table