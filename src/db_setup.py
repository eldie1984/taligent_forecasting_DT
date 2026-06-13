import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
import sys
import time

from tqdm.auto import tqdm
tqdm.pandas(desc="Processing rows")

# Setup de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def connect_to_db():
    """Conectar a PostgreSQL y retornar el engine."""

    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "taligent_user"),
        "password": os.getenv("DB_PASSWORD", "taligent_pass"),
        "database": os.getenv("DB_NAME", "taligent_db"),
    }

    connection_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    try:
        engine = create_engine(connection_string)
        logger.info("Conexión a la base de datos exitosa.")
        return engine
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise


def drop_tables(engine):
    """Elimina tablas existentes (para fresh start)."""
    logger.info("Eliminando tablas existentes (si existen)...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS taligent.predictions CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS taligent.scoring_dataset CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS taligent.average_liquor_prices_per_county CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS taligent.store_level_liquor_sales CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS taligent.county_level_liquor_sales CASCADE;"))
        conn.commit()
    logger.info("Tablas eliminadas.")


def create_tables(engine):
    """Crea tablas necesarias para el proyecto."""

    logger.info("Creando tablas...")

    with engine.connect() as conn:
        # Tabla para ventas de licor a nivel de condado
        create_county_level_sales_table = text("""
            CREATE TABLE taligent.county_level_liquor_sales (
                id SERIAL PRIMARY KEY,
                county VARCHAR(100) ,
                product VARCHAR(255) ,
                total_sales FLOAT,
                year INTEGER NOT NULL,
                county_population INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(create_county_level_sales_table)
        conn.commit()
        logger.info("Tabla 'taligent.county_level_liquor_sales' creada.")

        # Tabla para ventas de licor a nivel de tienda
        create_store_level_sales_table = text("""
            CREATE TABLE taligent.store_level_liquor_sales (
                id SERIAL PRIMARY KEY,
                store VARCHAR(255) ,
                product VARCHAR(255) ,
                total_sales FLOAT ,
                year INTEGER ,
                month INTEGER ,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(create_store_level_sales_table)
        conn.commit()
        logger.info("Tabla 'taligent.store_level_liquor_sales' creada.")

        # Tabla para precios promedio de licor por condado
        create_average_prices_table = text("""
            CREATE TABLE taligent.average_liquor_prices_per_county (
                id SERIAL PRIMARY KEY,
                county VARCHAR(100) ,
                year INTEGER ,
                month INTEGER ,
                category VARCHAR(255) ,
                average_price_per_liter FLOAT ,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(create_average_prices_table)
        conn.commit()
        logger.info("Tabla 'taligent.average_liquor_prices_per_county' creada.")

        create_predictions_table = text("""
            CREATE TABLE taligent.predictions (
                id SERIAL PRIMARY KEY,
                scoring_id INTEGER ,
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
            )
        """)

        conn.execute(create_predictions_table)
        conn.commit()
        logger.info("Tabla 'taligent.predictions' creada.")

    


def populate_county_level_liquor_sales(engine):
    """Popula la tabla county_level_liquor_sales agregando datos de ventas por condado"""
    
    logger.info("Populando tabla taligent.county_level_liquor_sales...")
    
    query = text("""
        INSERT INTO taligent.county_level_liquor_sales (county, product, total_sales, year, county_population)
        SELECT 
            ils.county_name AS county,
            ils.im_desc AS product,
            SUM(ils.sales_dollars) AS total_sales,
            EXTRACT(YEAR FROM ils.ordered_on)::INTEGER AS year,
            ape.estimate AS county_population
        FROM taligent.iowa_liquor_sales ils
        LEFT JOIN taligent.annual_population_estimates ape 
            ON ils.county_fips_code = ape.county_fips_code 
            AND EXTRACT(YEAR FROM ils.ordered_on)::INTEGER = ape.estimate_year
            AND ape.geography_type = 'County'
        GROUP BY 
            ils.county_name, 
            ils.im_desc, 
            EXTRACT(YEAR FROM ils.ordered_on),
            ape.estimate
        ORDER BY county, year, total_sales DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        conn.commit()
        logger.info(f"{result.rowcount} filas insertadas en county_level_liquor_sales")


def populate_store_level_liquor_sales(engine):
    """Popula la tabla store_level_liquor_sales agregando datos de ventas por tienda"""
    
    logger.info("Populando tabla taligent.store_level_liquor_sales...")
    
    query = text("""
        INSERT INTO taligent.store_level_liquor_sales (store, product, total_sales, year, month)
        SELECT 
            ils.store_name AS store,
            ils.im_desc AS product,
            SUM(ils.sales_dollars) AS total_sales,
            EXTRACT(YEAR FROM ils.ordered_on)::INTEGER AS year,
            EXTRACT(MONTH FROM ils.ordered_on)::INTEGER AS month
        FROM taligent.iowa_liquor_sales ils
        GROUP BY 
            ils.store_name, 
            ils.im_desc, 
            EXTRACT(YEAR FROM ils.ordered_on),
            EXTRACT(MONTH FROM ils.ordered_on)
        ORDER BY store, year, month, total_sales DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        conn.commit()
        logger.info(f"{result.rowcount} filas insertadas en taligent.store_level_liquor_sales")


def populate_average_liquor_prices_per_county(engine):
    """Popula la tabla average_liquor_prices_per_county calculando precios promedio por litro"""
    
    logger.info("Populando tabla taligent.average_liquor_prices_per_county...")
    
    query = text("""
        INSERT INTO taligent.average_liquor_prices_per_county (county, year, month, category, average_price_per_liter)
        SELECT 
            ils.county_name AS county,
            EXTRACT(YEAR FROM ils.ordered_on)::INTEGER AS year,
            EXTRACT(MONTH FROM ils.ordered_on)::INTEGER AS month,
            ils.category_name AS category,
            AVG(CASE WHEN ils.sales_liters > 0 
                THEN ils.sales_dollars / ils.sales_liters 
                ELSE NULL END) AS average_price_per_liter
        FROM taligent.iowa_liquor_sales ils
        WHERE ils.sales_liters > 0
        GROUP BY 
            ils.county_name, 
            EXTRACT(YEAR FROM ils.ordered_on),
            EXTRACT(MONTH FROM ils.ordered_on),
            ils.category_name
        ORDER BY county, year, month, category
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        conn.commit()
        logger.info(f"{result.rowcount} filas insertadas en taligent.average_liquor_prices_per_county")
def is_recently_modified(file_path, max_age_seconds=60):
    """Check if a file was modified within the specified time window."""
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
        
    # Get the last modification time (timestamp)
    file_mtime = os.path.getmtime(file_path)
    current_time = time.time()
    
    # Calculate how many seconds ago it changed
    age_seconds = current_time - file_mtime
    
    return age_seconds <= max_age_seconds


def table_exists(engine, table_name, schema="taligent"):
    """Check if a table exists in the database."""
    query = text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{schema}' 
            AND table_name = '{table_name}'
        );
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        return result.scalar()


def truncate_aggregated_tables(engine):
    """Truncate the aggregated tables to prepare for repopulation."""
    logger.info("Truncando tablas agregadas...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE taligent.county_level_liquor_sales CASCADE;"))
        conn.execute(text("TRUNCATE TABLE taligent.store_level_liquor_sales CASCADE;"))
        conn.execute(text("TRUNCATE TABLE taligent.average_liquor_prices_per_county CASCADE;"))
        conn.commit()
    logger.info("Tablas agregadas truncadas.")


def truncate_original_tables(engine):
    """Truncate the original source tables to prepare for repopulation."""
    logger.info("Truncando tablas originales...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE taligent.iowa_liquor_sales CASCADE;"))
        conn.execute(text("TRUNCATE TABLE taligent.annual_population_estimates CASCADE;"))
        conn.commit()
    logger.info("Tablas originales truncadas.")


def repopulate_iowa_liquor_sales(engine, csv_path="data/iowa_liquor_sales_january_2012_current_1051_rows.csv"):
    """Repopulate the iowa_liquor_sales table from CSV file."""
    
    logger.info(f"Repopulando tabla taligent.iowa_liquor_sales desde {csv_path}...")
    
    # Verify file exists
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "iowa_liquor_sales_january_2012_current_1051_rows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Archivo CSV no encontrado en {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # Validate expected columns
    expected_columns = [
        "invoice_id", "ordered_on", "store_no", "store_name", "store_address",
        "store_city", "store_zip_code", "county_fips_code", "county_name",
        "category_code", "category_name", "vendor_number", "vendor_name",
        "item_no", "im_desc", "pack", "bottle_volume_ml", "sales_bottles",
        "sales_dollars", "sales_liters", "sales_gallons"
    ]
    missing_columns = set(expected_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")
    
    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Se encontraron {duplicate_count} filas duplicadas.")
        df = df.drop_duplicates()
        logger.info(f"Filas duplicadas eliminadas. Nuevo tamaño: {df.shape[0]} filas.")
    
    # Insert into database
    logger.info("Insertando datos en la base de datos...")
    df[expected_columns].to_sql(
        "iowa_liquor_sales", con=engine, if_exists="append", index=False, schema="taligent"
    )
    
    logger.info(f"{len(df)} datos insertados exitosamente en taligent.iowa_liquor_sales.")


def repopulate_annual_population_estimates(engine, csv_path="data/annual_population_estimates_for_state_of_iowa_cities_and_counties_1101_rows.csv"):
    """Repopulate the annual_population_estimates table from CSV file."""
    
    logger.info(f"Repopulando tabla taligent.annual_population_estimates desde {csv_path}...")
    
    # Verify file exists
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "annual_population_estimates_for_state_of_iowa_cities_and_counties_1101_rows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Archivo CSV no encontrado en {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset de población cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # Validate expected columns
    expected_columns = [
        "record_id", "fips_code", "state_fips_code", "county_fips_code", "city_fips_code",
        "geography_type", "geography_area", "estimate_type", "estimate_year", "estimate",
        "change", "change_rate", "estimate_as_of", "city_latitude", "city_longitude",
        "publication", "reference_file", "release_date", "is_current_annual_est"
    ]
    missing_columns = set(expected_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset de población: {missing_columns}")
    
    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Se encontraron {duplicate_count} filas duplicadas.")
        df = df.drop_duplicates()
        logger.info(f"Filas duplicadas eliminadas. Nuevo tamaño: {df.shape[0]} filas.")
    
    # Insert into database
    logger.info("Insertando datos de población en la base de datos...")
    df[expected_columns].to_sql(
        "annual_population_estimates", con=engine, if_exists="append", index=False, schema="taligent"
    )
    
    logger.info(f"{len(df)} datos de población insertados exitosamente en taligent.annual_population_estimates.")

def main():
    """Pipeline principal para setup de la base de datos."""

    try:
        logger.info("Iniciando setup de la base de datos...")
        engine = connect_to_db()

        # Check if aggregated tables exist
        county_table_exists = table_exists(engine, "county_level_liquor_sales")
        store_table_exists = table_exists(engine, "store_level_liquor_sales")
        prices_table_exists = table_exists(engine, "average_liquor_prices_per_county")
        
        aggregated_tables_exist = county_table_exists and store_table_exists and prices_table_exists
        
        # Define the CSV file paths to check for modifications
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_to_check_population = os.path.join(script_dir, "..", "data", "annual_population_estimates_for_state_of_iowa_cities_and_counties_1101_rows.csv")
        file_to_check_sales = os.path.join(script_dir, "..", "data", "iowa_liquor_sales_january_2012_current_1051_rows.csv")
        
        # Check if either CSV file has been modified recently (within 60 seconds)
        csv_modified = is_recently_modified(file_to_check_population, max_age_seconds=60) or is_recently_modified(file_to_check_sales, max_age_seconds=60)
        
        # If aggregated tables don't exist, create and populate them
        if not aggregated_tables_exist:
            logger.info("Tablas agregadas no existen. Creando y poblando tablas...")
            create_tables(engine)
            logger.info("Iniciando data pipeline para tablas agregadas...")
            populate_county_level_liquor_sales(engine)
            populate_store_level_liquor_sales(engine)
            populate_average_liquor_prices_per_county(engine)
            logger.info("Tablas agregadas creadas y pobladas exitosamente.")
        # If CSV files have been modified, repopulate all tables
        elif csv_modified:
            logger.info("CSV files have been modified. Repopulating all tables...")
            
            # Repopulate original tables
            truncate_original_tables(engine)
            repopulate_iowa_liquor_sales(engine)
            repopulate_annual_population_estimates(engine)
            
            # Repopulate aggregated tables
            truncate_aggregated_tables(engine)
            logger.info("Iniciando data pipeline para tablas agregadas...")
            populate_county_level_liquor_sales(engine)
            populate_store_level_liquor_sales(engine)
            populate_average_liquor_prices_per_county(engine)
            logger.info("Todas las tablas repobladas exitosamente.")
        else:
            logger.info("Tablas agregadas existen y CSV files have not been modified recently. No action needed.")
        
        logger.info("Setup de la base de datos completado exitosamente.")
        return True

    except Exception as e:
        logger.error(f"Error durante el setup de la base de datos: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
