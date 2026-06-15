# Taligent Data Engineering Challenge

A data engineering project focused on building a robust data pipeline for Iowa liquor sales analytics and predictive modeling using public BigQuery datasets.

## Overview

This project processes and analyzes Iowa liquor retail sales data combined with census population data to create derivative datasets for reporting and machine learning. The solution leverages:

- **Google BigQuery** - Public datasets for Iowa liquor sales and census data
- **PostgreSQL** - Data warehouse for processed datasets
- **Python** - Data engineering, feature engineering, and ML pipeline
- **Docker** - Containerized infrastructure
- **MLOps** - Model training, evaluation, and scheduling

## Project Goals

### Core Deliverables

1. **Data Pipeline** - ETL process combining Iowa liquor sales and census data
2. **Derived Datasets** - Create three new analytical tables:
   - County-level liquor sales with population and totals
   - Store-level sales with monthly granularity
   - Average liquor prices per county by category
3. **Predictive Model** - Sales forecast model with feature engineering and hyperparameter tuning
4. **Scheduling & Automation** - Weekly scheduled model runs with CI/CD integration

### Bonus Features

- Automatic pipeline updates on source data changes
- Cloud deployment (Cloud Run or similar)
- Sales assistant agent for model interpretation

## Prerequisites

- Python 3.13+
- Docker & Docker Compose
- Google Cloud account with BigQuery access
- PostgreSQL (or use Docker)

## Project Structure

```
taligent/DE/
├── src/
│   ├── __init__.py
│   ├── db_setup.py           # Database initialization and schema
│   ├── training.py           # ML model training pipeline
│   └── utils.py              # Shared utilities and helpers
├── database/
│   └── init.sql              # PostgreSQL schema initialization
├── data/
│   ├── iowa_liquor_sales_*.csv
│   └── annual_population_estimates_*.csv
├── docker-compose.yaml       # Container orchestration
├── Dockerfile               # Python environment setup
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd taligent/DE
```

### 2. Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure BigQuery Access

```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate with GCP
gcloud auth application-default login

# Set your GCP project ID
export GCP_PROJECT_ID=<your-project-id>
```

### 4. Set Environment Variables

Create a `.env` file:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=taligent_db
DB_USER=taligent_user
DB_PASSWORD=taligent_pass

# GCP Configuration
GCP_PROJECT_ID=<your-project-id>
GCP_DATASET_ID=iowa_liquor_sale

# Application Configuration
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

## Usage

### Option 1: Using Docker Compose (Recommended)

```bash
# Start all services (PostgreSQL, ML Pipeline)
docker-compose up -d

# View logs
docker-compose logs -f me_pipeline

# Stop services
docker-compose down
```

### Option 2: Local Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run database setup
python -m src.db_setup

# Run training pipeline
python -m src.training

# Run tests
pytest tests/ -v --cov=src
```

## Data Flow

### 1. Data Ingestion

- **Iowa Liquor Sales**: Public BigQuery dataset
  - Source: `bigquery-public-data.iowa_liquor_sale.sales`
  - Attributes: Date, store, product, county, quantity, price

- **Census Data**: Iowa population estimates
  - Source: CSV files in `/data` directory
  - Attributes: County, year, population

### 2. Data Processing

The pipeline creates three derivative datasets:

#### County-Level Sales
```sql
SELECT
    county,
    product,
    SUM(sale_amount) as total_sales,
    YEAR(date) as year,
    population
FROM sales
GROUP BY county, product, year
```

#### Store-Level Sales
```sql
SELECT
    store_id,
    store_name,
    product,
    SUM(sale_amount) as total_sales,
    YEAR(date) as year,
    MONTH(date) as month
FROM sales
GROUP BY store_id, store_name, product, year, month
```

#### Average Prices by County
```sql
SELECT
    county,
    product_category,
    YEAR(date) as year,
    MONTH(date) as month,
    AVG(price_per_liter) as avg_price
FROM sales
GROUP BY county, product_category, year, month
```

### 3. Feature Engineering

The model pipeline includes:

- **Temporal features**: Year, month, quarter, day of week, holidays
- **Aggregation features**: Rolling averages, growth rates, seasonal decomposition
- **Geographic features**: County-level statistics, population trends
- **Categorical encoding**: One-hot encoding for products and counties

### 4. Model Training

The training pipeline:

1. Loads processed data from PostgreSQL
2. Applies feature engineering transformations
3. Performs feature selection (correlation analysis, feature importance)
4. Hyperparameter tuning using cross-validation
5. Model evaluation with train/test split
6. Model persistence for deployment

Supported models:
- Linear Regression (baseline)
- Random Forest
- XGBoost
- ARIMA (time-series)

## Configuration

### Database Schema

The PostgreSQL database includes tables for:

- `iowa_liquor_sales` - Raw sales transactions
- `iowa_population` - County population by year
- `county_sales_aggregated` - Derived county-level metrics
- `store_sales_aggregated` - Derived store-level metrics
- `county_avg_prices` - Derived price analytics
- `model_predictions` - Forecast results

### Scheduling

Weekly model runs are configured via Ofelia (Docker task scheduler):

```ini
# ofelia.ini
[job-exec "weekly-model-training"]
schedule = @weekly
container = taligent_me_pipeline
command = python -m src.training --weekly
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

### Adding New Features

1. Write tests first (TDD approach)
2. Implement feature in `src/`
3. Update `db_setup.py` if schema changes are needed
4. Run full test suite
5. Update this README with new capabilities

## Deployment

### Google Cloud Run

```bash
# Build container
docker build -t gcr.io/$GCP_PROJECT_ID/taligent-pipeline .

# Push to Container Registry
docker push gcr.io/$GCP_PROJECT_ID/taligent-pipeline

# Deploy to Cloud Run
gcloud run deploy taligent-pipeline \
  --image gcr.io/$GCP_PROJECT_ID/taligent-pipeline \
  --platform managed \
  --region us-central1 \
  --set-env-vars DB_HOST=<cloud-sql-ip>,DB_PORT=5432
```

## Performance Considerations

- **Data Volume**: Iowa liquor dataset contains ~1M+ rows; optimize queries with indexes
- **BigQuery Pricing**: Consider caching frequently accessed datasets locally
- **Model Training**: Use sample data for development; full dataset for production
- **Storage**: PostgreSQL data volumes grow with historical data; implement retention policies

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps

# View PostgreSQL logs
docker-compose logs postgres

# Test connection
psql -h localhost -U taligent_user -d taligent_db
```

### BigQuery Authentication

```bash
# Verify authentication
gcloud auth list

# Re-authenticate if needed
gcloud auth application-default login
```

### Model Training Failures

```bash
# Check logs
docker-compose logs me_pipeline

# Run with verbose output
python -m src.training --verbose
```

## Key Dependencies

- **google-cloud-bigquery** - BigQuery API access
- **sqlalchemy** - Database ORM
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning models
- **xgboost** - Gradient boosting
- **statsmodels** - Time-series analysis
- **fastapi** - API framework (for future Sales Assistant)

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Write tests and implementation
3. Ensure all tests pass: `pytest`
4. Commit with meaningful message: `git commit -m "feat: add your feature"`
5. Push and create Pull Request

## Next Steps

- [ ] Implement data ingestion from BigQuery
- [ ] Create database schema in PostgreSQL
- [ ] Build feature engineering pipeline
- [ ] Train and evaluate predictive models
- [ ] Deploy to Cloud Run with scheduling
- [ ] Build Sales Assistant agent (Langchain/OpenAI)
- [ ] Set up CI/CD pipeline
- [ ] Add comprehensive monitoring and alerting

## Resources

- [Iowa Liquor Sales Dataset](https://console.cloud.google.com/projectselector2/bigquery?p=bigquery-public-data&d=iowa_liquor_sale)
- [Iowa Data Portal](https://data.iowa.gov/)
- [Google BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## License

This project is part of the Taligent technical challenge.

## Contact

For questions or issues, please refer to the project maintainers.
