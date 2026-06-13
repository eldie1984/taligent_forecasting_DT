-- Taligent Iowa Liquor Sales Database Schema
-- This script sets up the database schema for the ML pipeline

-- Create extension for UUID generation (if needed)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for the application
CREATE SCHEMA IF NOT EXISTS taligent;


-- Iowa Liquor Sales Table
CREATE TABLE IF NOT EXISTS taligent.iowa_liquor_sales (
    id SERIAL PRIMARY KEY,
    invoice_id VARCHAR(50) NOT NULL,
    ordered_on DATE NOT NULL,
    store_no VARCHAR(50) NOT NULL,
    store_name VARCHAR(255) NOT NULL,
    store_address VARCHAR(255),
    store_city VARCHAR(100),
    store_zip_code VARCHAR(20),
    county_fips_code VARCHAR(20),
    county_name VARCHAR(100),
    category_code VARCHAR(50),
    category_name VARCHAR(255),
    vendor_number VARCHAR(50),
    vendor_name VARCHAR(255),
    item_no VARCHAR(50),
    im_desc VARCHAR(255),
    pack INTEGER,
    bottle_volume_ml FLOAT,
    sales_bottles FLOAT,
    sales_dollars FLOAT,
    sales_liters FLOAT,
    sales_gallons FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Annual Population Estimates Table
CREATE TABLE IF NOT EXISTS taligent.annual_population_estimates (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(50) NOT NULL,
    fips_code VARCHAR(20),
    state_fips_code VARCHAR(20),
    county_fips_code VARCHAR(20),
    city_fips_code VARCHAR(20),
    geography_type VARCHAR(50),
    geography_area VARCHAR(255),
    estimate_type VARCHAR(100),
    estimate_year INTEGER,
    estimate INTEGER,
    change INTEGER,
    change_rate FLOAT,
    estimate_as_of DATE,
    city_latitude FLOAT,
    city_longitude FLOAT,
    publication VARCHAR(255),
    reference_file VARCHAR(255),
    release_date VARCHAR(100),
    is_current_annual_est BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions Table
CREATE TABLE IF NOT EXISTS taligent.predictions (
    id SERIAL PRIMARY KEY,
    scoring_id INTEGER NOT NULL,
    invoice_id VARCHAR(50),
    ordered_on DATE,
    store_no VARCHAR(50),
    store_name VARCHAR(255),
    category_name VARCHAR(255),
    vendor_name VARCHAR(255),
    im_desc VARCHAR(255),
    pack INTEGER,
    bottle_volume_ml FLOAT,
    actual_sales_dollars FLOAT,
    predicted_sales_dollars FLOAT,
    absolute_error FLOAT,
    percentage_error FLOAT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_iowa_liquor_sales_invoice_id ON taligent.iowa_liquor_sales(invoice_id);
CREATE INDEX IF NOT EXISTS idx_iowa_liquor_sales_ordered_on ON taligent.iowa_liquor_sales(ordered_on);
CREATE INDEX IF NOT EXISTS idx_iowa_liquor_sales_store_no ON taligent.iowa_liquor_sales(store_no);
CREATE INDEX IF NOT EXISTS idx_iowa_liquor_sales_county_name ON taligent.iowa_liquor_sales(county_name);
CREATE INDEX IF NOT EXISTS idx_iowa_liquor_sales_category_name ON taligent.iowa_liquor_sales(category_name);

CREATE INDEX IF NOT EXISTS idx_population_estimates_record_id ON taligent.annual_population_estimates(record_id);
CREATE INDEX IF NOT EXISTS idx_population_estimates_geography_area ON taligent.annual_population_estimates(geography_area);
CREATE INDEX IF NOT EXISTS idx_population_estimates_estimate_year ON taligent.annual_population_estimates(estimate_year);
CREATE INDEX IF NOT EXISTS idx_population_estimates_geography_type ON taligent.annual_population_estimates(geography_type);

CREATE INDEX IF NOT EXISTS idx_predictions_scoring_id ON taligent.predictions(scoring_id);
CREATE INDEX IF NOT EXISTS idx_predictions_invoice_id ON taligent.predictions(invoice_id);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_time ON taligent.predictions(prediction_time);

-- Grant permissions to the taligent_user
GRANT ALL PRIVILEGES ON SCHEMA taligent TO taligent_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA taligent TO taligent_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA taligent TO taligent_user;

-- Set default search path to include taligent schema
ALTER DATABASE taligent_db SET search_path TO public, taligent;

COMMENT ON SCHEMA taligent IS 'Schema for Taligent Iowa Liquor Sales ML Pipeline';
COMMENT ON TABLE taligent.iowa_liquor_sales IS 'Iowa liquor sales transaction data';
COMMENT ON TABLE taligent.annual_population_estimates IS 'Annual population estimates for Iowa cities and counties';
COMMENT ON TABLE taligent.predictions IS 'ML model predictions for liquor sales';



COPY taligent.iowa_liquor_sales (invoice_id,ordered_on,store_no,store_name,store_address,store_city,store_zip_code,county_fips_code,county_name,category_code,category_name,vendor_number,vendor_name,item_no,im_desc,pack,bottle_volume_ml,sales_bottles,sales_dollars,sales_liters,sales_gallons) FROM '/data/iowa_liquor_sales.csv' DELIMITER ',' CSV HEADER;
COPY taligent.annual_population_estimates (record_id,fips_code,state_fips_code,county_fips_code,city_fips_code,geography_type,geography_area,estimate_type,estimate_year,estimate,change,change_rate,estimate_as_of,city_latitude,city_longitude,publication,reference_file,release_date,is_current_annual_est) FROM '/data/annual_population_estimates.csv' DELIMITER ',' CSV HEADER;