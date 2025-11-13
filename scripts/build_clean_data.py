from pathlib import Path
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(exist_ok=True)


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return value
    text = str(value).strip().upper()
    normalized = ''.join(
        ch for ch in unicodedata.normalize('NFD', text)
        if unicodedata.category(ch) != 'Mn'
    )
    return normalized.replace(' ', '_')


def mode_or_unknown(series: pd.Series) -> str:
    mode = series.mode()
    if mode.empty:
        return 'UNKNOWN'
    return mode.iloc[0]


def build_geolocation_lookup(geo_path: Path) -> pd.DataFrame:
    geolocation = pd.read_csv(geo_path)
    lookup = (
        geolocation.groupby('geolocation_zip_code_prefix', as_index=False)
        .agg(
            latitude=('geolocation_lat', 'median'),
            longitude=('geolocation_lng', 'median')
        )
    )
    return lookup


def main() -> None:
    orders = pd.read_csv(
        RAW_DIR / 'olist_orders_dataset.csv',
        parse_dates=[
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    )
    order_items = pd.read_csv(
        RAW_DIR / 'olist_order_items_dataset.csv',
        parse_dates=['shipping_limit_date']
    )
    customers = pd.read_csv(RAW_DIR / 'olist_customers_dataset.csv')
    sellers = pd.read_csv(RAW_DIR / 'olist_sellers_dataset.csv')
    geolocation_lookup = build_geolocation_lookup(RAW_DIR / 'olist_geolocation_dataset.csv')
    payments = pd.read_csv(RAW_DIR / 'olist_order_payments_dataset.csv')
    reviews = pd.read_csv(RAW_DIR / 'olist_order_reviews_dataset.csv')
    products = pd.read_csv(RAW_DIR / 'olist_products_dataset.csv')
    category_translation = pd.read_csv(RAW_DIR / 'product_category_name_translation.csv')

    # Entrega real como variável alvo
    orders = orders[orders['order_delivered_customer_date'].notna()].copy()
    orders['delivery_time_days'] = (
        (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp'])
        .dt.total_seconds()
        / 86400
    )
    orders['approval_delay_hours'] = (
        (orders['order_approved_at'] - orders['order_purchase_timestamp'])
        .dt.total_seconds()
        / 3600
    )
    orders['delivery_estimate_gap_days'] = (
        (orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date'])
        .dt.total_seconds()
        / 86400
    )
    orders = orders[orders['delivery_time_days'] >= 0]

    # Informações de itens/pedidos
    category_translation = category_translation.fillna('UNKNOWN')
    products = products.merge(
        category_translation,
        on='product_category_name',
        how='left'
    )
    products['product_category_name_english'] = (
        products['product_category_name_english']
        .fillna('UNKNOWN')
        .astype(str)
    )
    order_items = order_items.merge(
        products[['product_id', 'product_category_name_english']],
        on='product_id',
        how='left'
    )

    items_agg = (
        order_items.groupby('order_id', as_index=False)
        .agg(
            items_count=('order_item_id', 'count'),
            unique_sellers=('seller_id', 'nunique'),
            unique_products=('product_id', 'nunique'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
        )
    )
    items_agg['price_per_item'] = items_agg['total_price'] / items_agg['items_count']

    category_mode = (
        order_items.groupby('order_id')
        .agg(primary_category=('product_category_name_english', mode_or_unknown))
        .reset_index()
    )

    payment_agg = (
        payments.groupby('order_id', as_index=False)
        .agg(
            payment_value_sum=('payment_value', 'sum'),
            payment_installments_max=('payment_installments', 'max'),
            payment_type=('payment_type', 'first')
        )
    )

    review_agg = (
        reviews.groupby('order_id', as_index=False)
        .agg(
            average_review_score=('review_score', 'mean'),
            review_count=('review_id', 'count')
        )
    )

    # Integração com clientes
    customers = customers.merge(
        geolocation_lookup,
        left_on='customer_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    customers.rename(
        columns={
            'latitude': 'customer_latitude',
            'longitude': 'customer_longitude'
        },
        inplace=True
    )

    # Integração com sellers
    sellers = sellers.merge(
        geolocation_lookup,
        left_on='seller_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    sellers.rename(
        columns={
            'latitude': 'seller_latitude',
            'longitude': 'seller_longitude'
        },
        inplace=True
    )

    seller_for_order = (
        order_items.sort_values('order_item_id')
        .drop_duplicates('order_id')
        .merge(
            sellers[['seller_id', 'seller_city', 'seller_state', 'seller_latitude', 'seller_longitude']],
            on='seller_id',
            how='left'
        )
        .rename(
            columns={
                'seller_city': 'primary_seller_city',
                'seller_state': 'primary_seller_state'
            }
        )
        [[
            'order_id',
            'primary_seller_city',
            'primary_seller_state',
            'seller_latitude',
            'seller_longitude'
        ]]
    )

    dataset = (
        orders
        .merge(items_agg, on='order_id', how='left')
        .merge(category_mode, on='order_id', how='left')
        .merge(payment_agg, on='order_id', how='left')
        .merge(review_agg, on='order_id', how='left')
        .merge(
            customers[['customer_id', 'customer_city', 'customer_state', 'customer_latitude', 'customer_longitude']],
            on='customer_id',
            how='left'
        )
        .merge(
            seller_for_order,
            on='order_id',
            how='left'
        )
    )

    dataset['order_month'] = dataset['order_purchase_timestamp'].dt.month
    dataset['order_weekday'] = dataset['order_purchase_timestamp'].dt.dayofweek

    # Usa geopandas para calcular distância real entre supplier e cliente
    dataset = dataset.dropna(subset=[
        'customer_latitude',
        'customer_longitude',
        'seller_latitude',
        'seller_longitude'
    ]).copy()
    print(f"Dataset após remoção de nulos: {dataset.shape}")
    geo_df = gpd.GeoDataFrame(
        dataset,
        geometry=gpd.points_from_xy(dataset['customer_longitude'], dataset['customer_latitude']),
        crs='EPSG:4326'
    )
    seller_geo = gpd.GeoDataFrame(
        dataset,
        geometry=gpd.points_from_xy(dataset['seller_longitude'], dataset['seller_latitude']),
        crs='EPSG:4326'
    )
    dataset['distance_km'] = (
        geo_df.geometry.to_crs('EPSG:3857')
        .distance(seller_geo.geometry.to_crs('EPSG:3857'))
        / 1000
    )

    categorical_columns = [
        'customer_state',
        'primary_seller_state',
        'payment_type',
        'primary_category'
    ]
    dataset.drop(columns=['customer_city', 'primary_seller_city'], inplace=True)
    for column in categorical_columns:
        dataset[column] = dataset[column].apply(normalize_text)

    numeric_columns = dataset.select_dtypes(include='number').columns.tolist()
    target_column = 'delivery_time_days'
    for column in ['order_id', 'customer_id', 'order_status']:
        if column in numeric_columns:
            numeric_columns.remove(column)
    numeric_columns.remove(target_column)
    excluded_from_scaling = [
        'customer_latitude',
        'customer_longitude',
        'seller_latitude',
        'seller_longitude'
    ]
    scale_columns = [col for col in numeric_columns if col not in excluded_from_scaling]

    scaler = StandardScaler()
    dataset[scale_columns] = scaler.fit_transform(dataset[scale_columns])

    imputer = SimpleImputer(strategy='median')
    dataset[scale_columns] = imputer.fit_transform(dataset[scale_columns])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(dataset[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=dataset.index)

    final_dataset = (
        pd.concat([
            dataset.drop(columns=categorical_columns + ['order_status']),
            encoded_df
        ], axis=1)
        .reset_index(drop=True)
    )

    final_dataset.to_csv(CLEAN_DIR / 'olist_ml_ready.csv', index=False)
    print('Cleaned dataset gravado em data/clean/olist_ml_ready.csv')


if __name__ == '__main__':
    main()
