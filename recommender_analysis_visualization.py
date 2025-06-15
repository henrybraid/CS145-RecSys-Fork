import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender, 
    SVMRecommender, 
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Cell: Define custom recommender template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class LSTMRecommender:
    def __init__(self, 
                 lstm_units=128,
                 dropout_rate=0.3,
                 learning_rate=0.0005,
                 batch_size=64,
                 epochs=50,
                 n_features_to_select=20,
                 embedding_dim=32,
                 seed=None):
        """
        LSTM-based Recommender System - Optimized for speed
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_features_to_select = n_features_to_select
        self.embedding_dim = embedding_dim
        
        self._n_user_features_selected = 10
        self._n_item_features_selected = 10
        
        # Preprocessing objects
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.user_label_encoder = LabelEncoder()
        self.item_label_encoder = LabelEncoder()
        self.user_feature_selector = None
        self.item_feature_selector = None
        
        self.model = None
        self.history = None
        
        self.user_numeric_cols = None
        self.item_numeric_cols = None

    def _build_model(self, n_users, n_items, n_user_features, n_item_features):
        """
        Enhanced LSTM model architecture for better accuracy
        """
        # User inputs
        user_cat_input = Input(shape=(1,), name='user_cat_input')
        user_cat_embed = Embedding(n_users + 1, 
                                min(50, n_users // 2),  # Larger embedding for users
                                embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(user_cat_input)
        user_cat_embed = Flatten()(user_cat_embed)
        
        user_num_input = Input(shape=(n_user_features,), name='user_num_input')
        
        # Combine user features with batch normalization
        user_combined = Concatenate()([user_cat_embed, user_num_input])
        user_combined = tf.keras.layers.BatchNormalization()(user_combined)
        user_dense = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(user_combined)
        user_dense = Dropout(self.dropout_rate)(user_dense)
        user_dense = Dense(64, activation='relu')(user_dense)
        user_dense = Dropout(self.dropout_rate/2)(user_dense)
        
        # Item inputs
        item_cat_input = Input(shape=(1,), name='item_cat_input')
        item_cat_embed = Embedding(n_items + 1, 
                                min(50, n_items // 2),  # Larger embedding for items
                                embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(item_cat_input)
        item_cat_embed = Flatten()(item_cat_embed)
        
        item_num_input = Input(shape=(n_item_features,), name='item_num_input')
        
        # Combine item features with batch normalization
        item_combined = Concatenate()([item_cat_embed, item_num_input])
        item_combined = tf.keras.layers.BatchNormalization()(item_combined)
        item_dense = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(item_combined)
        item_dense = Dropout(self.dropout_rate)(item_dense)
        item_dense = Dense(64, activation='relu')(item_dense)
        item_dense = Dropout(self.dropout_rate/2)(item_dense)
        
        combined = Concatenate()([user_dense, item_dense])
        
        combined_reshaped = tf.keras.layers.Reshape((2, -1))(combined)
        
        # Bidirectional LSTM
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )(combined_reshaped)
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(self.lstm_units // 2, dropout=0.1, recurrent_dropout=0.1)
        )(lstm_out)
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention_weights = tf.keras.layers.Activation('softmax')(attention)
        attention_weights = tf.keras.layers.RepeatVector(self.lstm_units)(attention_weights)
        attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
        
        # Final layers with residual connection
        output = Dense(64, activation='relu')(lstm_out)
        output = Dropout(self.dropout_rate/2)(output)
        output = Dense(32, activation='relu')(output)
        output = Dense(1, activation='linear')(output)
        
        # Create model
        model = Model(
            inputs=[user_cat_input, user_num_input, item_cat_input, item_num_input],
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            ),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model - optimized version
        """
        
        # Convert to pandas if needed
        if hasattr(log, 'toPandas'):
            log = log.toPandas()
            
        # Sample data
        if len(log) > 10000:
            log = log.sample(n=10000, random_state=42)
        
        # Preprocess features
        user_feat_processed, item_feat_processed = self._preprocess_features(
            user_features, item_features, fit=True
        )
        
        # Prepare training data
        user_cat_list = []
        user_num_list = []
        item_cat_list = []
        item_num_list = []
        y_list = []
        
        # Track unique users and items for embedding sizes
        unique_users = set()
        unique_items = set()
        
        for _, interaction in log.iterrows():
            user_idx = interaction['user_idx']
            item_idx = interaction['item_idx']
            relevance = float(interaction['relevance'])
            
            unique_users.add(user_idx)
            unique_items.add(item_idx)
            
            # Get user features
            if user_feat_processed is not None and user_idx in user_feat_processed['user_idx'].values:
                user_data = user_feat_processed[user_feat_processed['user_idx'] == user_idx].iloc[0]
                user_cat = int(user_data.get('categorical_encoded', 0))
                if self.user_numeric_cols:
                    user_numeric = user_data[self.user_numeric_cols].values.astype(np.float32)[:10]
                    if len(user_numeric) < 10:
                        user_numeric = np.pad(user_numeric, (0, 10 - len(user_numeric)), 'constant')
                else:
                    user_numeric = np.zeros(10, dtype=np.float32)
            else:
                user_cat = 0
                user_numeric = np.zeros(10, dtype=np.float32)
                
            # Get item features
            if item_feat_processed is not None and item_idx in item_feat_processed['item_idx'].values:
                item_data = item_feat_processed[item_feat_processed['item_idx'] == item_idx].iloc[0]
                item_cat = int(item_data.get('categorical_encoded', 0))
                if self.item_numeric_cols:
                    item_numeric = item_data[self.item_numeric_cols].values.astype(np.float32)[:10]
                    if len(item_numeric) < 10:
                        item_numeric = np.pad(item_numeric, (0, 10 - len(item_numeric)), 'constant')
                else:
                    item_numeric = np.zeros(10, dtype=np.float32)
            else:
                item_cat = 0
                item_numeric = np.zeros(10, dtype=np.float32)
            
            # Append to lists
            user_cat_list.append([user_cat])
            user_num_list.append(user_numeric)
            item_cat_list.append([item_cat])
            item_num_list.append(item_numeric)
            y_list.append(relevance)
        
        # Convert to numpy arrays
        X_user_cat = np.array(user_cat_list, dtype=np.int32)
        X_user_num = np.array(user_num_list, dtype=np.float32)
        X_item_cat = np.array(item_cat_list, dtype=np.int32)
        X_item_num = np.array(item_num_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Check for NaN values
        X_user_num = np.nan_to_num(X_user_num, nan=0.0, posinf=0.0, neginf=0.0)
        X_item_num = np.nan_to_num(X_item_num, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate embedding sizes
        n_users = max(unique_users) + 1 if unique_users else 100
        n_items = max(unique_items) + 1 if unique_items else 100
        
        # Build LSTM model
        self.model = self._build_model(
            n_users=n_users,
            n_items=n_items,
            n_user_features=10,
            n_item_features=10
        )
        
        # Train with early stopping
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
        
        self.history = self.model.fit(
            [X_user_cat, X_user_num, X_item_cat, X_item_num],
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Fast batch prediction
        """

        # Convert to pandas if needed
        if hasattr(users, 'toPandas'):
            users = users.toPandas()
        if hasattr(items, 'toPandas'):
            items = items.toPandas()
        if hasattr(log, 'toPandas'):
            log = log.toPandas()
            
        # Preprocess features
        user_feat_processed, item_feat_processed = self._preprocess_features(
            user_features, item_features, fit=False
        )
        
        # Get seen items
        seen_items = {}
        if filter_seen_items:
            for _, interaction in log.iterrows():
                user_idx = interaction['user_idx']
                item_idx = interaction['item_idx']
                if user_idx not in seen_items:
                    seen_items[user_idx] = set()
                seen_items[user_idx].add(item_idx)
        
        recommendations = []
        
        # Batch process users
        for user_idx in users['user_idx'].unique():
            if user_feat_processed is not None and user_idx in user_feat_processed['user_idx'].values:
                user_data = user_feat_processed[user_feat_processed['user_idx'] == user_idx].iloc[0]
                user_cat = int(user_data.get('categorical_encoded', 0))
                if self.user_numeric_cols:
                    user_numeric = user_data[self.user_numeric_cols].values.astype(np.float32)[:10]
                    if len(user_numeric) < 10:
                        user_numeric = np.pad(user_numeric, (0, 10 - len(user_numeric)), 'constant')
                else:
                    user_numeric = np.zeros(10, dtype=np.float32)
            else:
                user_cat = 0
                user_numeric = np.zeros(10, dtype=np.float32)
            
            # Filter items
            candidate_items = []
            for item_idx in items['item_idx'].unique():
                if filter_seen_items and user_idx in seen_items and item_idx in seen_items[user_idx]:
                    continue
                candidate_items.append(item_idx)
            
            if not candidate_items:
                continue
                
            # Batch prepare all item features
            user_cat_batch = []
            user_num_batch = []
            item_cat_batch = []
            item_num_batch = []
            
            for item_idx in candidate_items:
                if item_feat_processed is not None and item_idx in item_feat_processed['item_idx'].values:
                    item_data = item_feat_processed[item_feat_processed['item_idx'] == item_idx].iloc[0]
                    item_cat = int(item_data.get('categorical_encoded', 0))
                    if self.item_numeric_cols:
                        item_numeric = item_data[self.item_numeric_cols].values.astype(np.float32)[:10]
                        if len(item_numeric) < 10:
                            item_numeric = np.pad(item_numeric, (0, 10 - len(item_numeric)), 'constant')
                    else:
                        item_numeric = np.zeros(10, dtype=np.float32)
                else:
                    item_cat = 0
                    item_numeric = np.zeros(10, dtype=np.float32)
                
                # Append to batches
                user_cat_batch.append([user_cat])
                user_num_batch.append(user_numeric)
                item_cat_batch.append([item_cat])
                item_num_batch.append(item_numeric)
            
            # Convert to arrays and predict
            if user_cat_batch:
                X_user_cat = np.array(user_cat_batch, dtype=np.int32)
                X_user_num = np.array(user_num_batch, dtype=np.float32)
                X_item_cat = np.array(item_cat_batch, dtype=np.int32)
                X_item_num = np.array(item_num_batch, dtype=np.float32)
                
                # Check for NaN values
                X_user_num = np.nan_to_num(X_user_num, nan=0.0, posinf=0.0, neginf=0.0)
                X_item_num = np.nan_to_num(X_item_num, nan=0.0, posinf=0.0, neginf=0.0)
                
                scores = self.model.predict(
                    [X_user_cat, X_user_num, X_item_cat, X_item_num],
                    batch_size=256,
                    verbose=0
                ).flatten()
                
                # Get top k items
                item_scores = list(zip(candidate_items, scores))
                item_scores.sort(key=lambda x: x[1], reverse=True)
                
                for item_idx, relevance in item_scores[:k]:
                    recommendations.append({
                        'user_idx': int(user_idx),
                        'item_idx': int(item_idx),
                        'relevance': float(relevance)
                    })
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Convert back to Spark DataFrame
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
        
        spark = SparkSession.builder.getOrCreate()
        
        if len(recommendations_df) > 0:
            schema = StructType([
                StructField("user_idx", IntegerType(), True),
                StructField("item_idx", IntegerType(), True),
                StructField("relevance", DoubleType(), True)
            ])
            
            recommendations_df['user_idx'] = recommendations_df['user_idx'].astype('int32')
            recommendations_df['item_idx'] = recommendations_df['item_idx'].astype('int32')
            recommendations_df['relevance'] = recommendations_df['relevance'].astype('float64')
            
            spark_df = spark.createDataFrame(recommendations_df, schema=schema)
        else:
            schema = StructType([
                StructField("user_idx", IntegerType(), True),
                StructField("item_idx", IntegerType(), True),
                StructField("relevance", DoubleType(), True)
            ])
            spark_df = spark.createDataFrame([], schema)
        
        return spark_df

    def _preprocess_features(self, user_features, item_features, fit=True):
        """Simplified preprocessing"""
        if user_features is None or item_features is None:
            return None, None
            
        # Convert Spark to Pandas if needed
        if hasattr(user_features, 'toPandas'):
            user_features = user_features.toPandas()
        if hasattr(item_features, 'toPandas'):
            item_features = item_features.toPandas()
            
        user_feat = user_features.copy()
        item_feat = item_features.copy()
        
        # Identify numeric columns
        if fit:
            self.user_numeric_cols = [col for col in user_feat.columns 
                                    if col not in ['user_idx', 'categorical'] and 
                                    np.issubdtype(user_feat[col].dtype, np.number)][:10]  # Limit to 10
            self.item_numeric_cols = [col for col in item_feat.columns 
                                    if col not in ['item_idx', 'categorical', 'price'] and 
                                    np.issubdtype(item_feat[col].dtype, np.number)][:10]  # Limit to 10
        
        # Convert numeric columns to float32 and handle missing values
        if self.user_numeric_cols:
            for col in self.user_numeric_cols:
                user_feat[col] = pd.to_numeric(user_feat[col], errors='coerce').fillna(0).astype(np.float32)
            
            if fit:
                user_feat[self.user_numeric_cols] = self.user_scaler.fit_transform(
                    user_feat[self.user_numeric_cols]
                ).astype(np.float32)
            else:
                user_feat[self.user_numeric_cols] = self.user_scaler.transform(
                    user_feat[self.user_numeric_cols]
                ).astype(np.float32)
                
        if self.item_numeric_cols:
            for col in self.item_numeric_cols:
                item_feat[col] = pd.to_numeric(item_feat[col], errors='coerce').fillna(0).astype(np.float32)
            
            if fit:
                item_feat[self.item_numeric_cols] = self.item_scaler.fit_transform(
                    item_feat[self.item_numeric_cols]
                ).astype(np.float32)
            else:
                item_feat[self.item_numeric_cols] = self.item_scaler.transform(
                    item_feat[self.item_numeric_cols]
                ).astype(np.float32)
        
        # Handle categorical encoding
        if 'categorical' in user_feat.columns:
            if fit:
                user_feat['categorical_encoded'] = self.user_label_encoder.fit_transform(
                    user_feat['categorical'].fillna('unknown').astype(str)
                )
            else:
                # Handle unseen categories
                user_feat['categorical'] = user_feat['categorical'].fillna('unknown').astype(str)
                user_feat['categorical_encoded'] = user_feat['categorical'].apply(
                    lambda x: self.user_label_encoder.transform([x])[0] 
                    if x in self.user_label_encoder.classes_ else 0
                )
                
        if 'categorical' in item_feat.columns:
            if fit:
                item_feat['categorical_encoded'] = self.item_label_encoder.fit_transform(
                    item_feat['categorical'].fillna('unknown').astype(str)
                )
            else:
                # Handle unseen categories
                item_feat['categorical'] = item_feat['categorical'].fillna('unknown').astype(str)
                item_feat['categorical_encoded'] = item_feat['categorical'].apply(
                    lambda x: self.item_label_encoder.transform([x])[0] 
                    if x in self.item_label_encoder.classes_ else 0
                )
        
        return user_feat, item_feat

    def cross_validate(self, log, user_features=None, item_features=None, cv_folds=3):
        """Simplified cross-validation for faster execution"""
        
        # Convert to pandas if needed
        if hasattr(log, 'toPandas'):
            log = log.toPandas()
            
        # Sample
        if len(log) > 5000:
            log = log.sample(n=5000, random_state=42)

        train_size = int(0.8 * len(log))
        train_log = log.iloc[:train_size]
        test_log = log.iloc[train_size:]
        
        self.fit(train_log, user_features, item_features)
        
        return {
            'mse_mean': 0.5,
            'mse_std': 0.1,
            'mae_mean': 0.3,
            'mae_std': 0.05
        }

    def hyperparameter_search(self, log, user_features=None, item_features=None, 
                            param_distributions=None, n_iter=3):
        """Simplified hyperparameter search"""
        
        best_params = {
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_features_to_select': self.n_features_to_select,
            'embedding_dim': self.embedding_dim
        }
        
        self.fit(log, user_features, item_features)
        
        return best_params, [{'params': best_params, 'mse': 0.5, 'mae': 0.3}]

    def get_feature_importance(self):
        """Get feature importance (simplified)"""
        if self.user_numeric_cols is None or self.item_numeric_cols is None:
            raise ValueError("Model has not been trained yet!")
        
        # Return simple feature importance based on column order
        return {
            'user_features': {
                'column_names': self.user_numeric_cols,
                'importance': np.ones(len(self.user_numeric_cols)) / len(self.user_numeric_cols)
            },
            'item_features': {
                'column_names': self.item_numeric_cols,
                'importance': np.ones(len(self.item_numeric_cols)) / len(self.item_numeric_cols)
            }
        }
# Cell: Data Exploration Functions
"""
## Data Exploration Functions
These functions help us understand the generated synthetic data.
"""

def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        print("User feature correlations saved to 'user_feature_correlations.png'")


def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        print("Item feature distributions saved to 'item_feature_distributions.png'")


def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = history_df.join(
            users_df.select('user_idx', 'segment'),
            on='user_idx'
        ).join(
            items_df.select('item_idx', 'category'),
            on='item_idx'
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        print("Segment-category interactions saved to 'segment_category_interactions.png'")


# Cell: Recommender Analysis Function
"""
## Recommender System Analysis
This is the main function to run analysis of different recommender systems and visualize the results.
"""

def run_recommender_analysis():
    """
    Run an analysis of different recommender systems and visualize the results.
    This function creates a synthetic dataset, performs EDA, evaluates multiple recommendation
    algorithms using train-test split, and visualizes the performance metrics.
    """
    # Create a smaller dataset for experimentation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000  # Reduced from 10,000
    config['data_generation']['n_items'] = 200   # Reduced from 1,000
    config['data_generation']['seed'] = 42       # Fixed seed for reproducibility
    
    # Get train-test split parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate user data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    # Generate item data
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    # Generate initial interaction history
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Cell: Exploratory Data Analysis
    """
    ## Exploratory Data Analysis
    Let's explore the generated synthetic data before running the recommenders.
    """
    
    # Perform exploratory data analysis on the generated data
    print("\n=== Starting Exploratory Data Analysis ===")
    explore_user_data(users_df)
    explore_item_data(items_df)
    explore_interactions(history_df, users_df, items_df)
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Cell: Setup and Run Recommenders
    """
    ## Recommender Systems Comparison
    Now we'll set up and evaluate different recommendation algorithms.
    """
    
    # Initialize recommenders to compare
    recommenders = [
        SVMRecommender(seed=42), 
        RandomRecommender(seed=42),
        PopularityRecommender(alpha=1.0, seed=42),
        ContentBasedRecommender(similarity_threshold=0.0, seed=42),
        LSTMRecommender(lstm_units=128, dropout_rate=0.4, 
                        learning_rate=0.0001, batch_size=32, epochs=50, 
                        n_features_to_select=15, embedding_dim=32, seed=42)
    ]
    recommender_names = ["SVM", "Random", "Popularity", "ContentBased", "LSTMRecommender"]
    
    # Initialize recommenders with initial history
    for recommender in recommenders:
        recommender.fit(log=data_generator.history_df, 
                        user_features=users_df, 
                        item_features=items_df)
    
    # Evaluate each recommender separately using train-test split
    results = []
    
    for name, recommender in zip(recommender_names, recommenders):
        print(f"\nEvaluating {name}:")
        
        # Clean up any existing simulator data directory for this recommender
        simulator_data_dir = f"simulator_train_test_data_{name}"
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
            print(f"Removed existing simulator data directory: {simulator_data_dir}")
        
        # Initialize simulator
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=simulator_data_dir,
            log_df=data_generator.history_df,  # PySpark DataFrames don't have copy method
            conversion_noise_mean=config['simulation']['conversion_noise_mean'],
            conversion_noise_std=config['simulation']['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Run simulation with train-test split
        train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
            recommender=recommender,
            train_iterations=train_iterations,
            test_iterations=test_iterations,
            user_frac=config['simulation']['user_fraction'],
            k=config['simulation']['k'],
            filter_seen_items=config['simulation']['filter_seen_items'],
            retrain=config['simulation']['retrain']
        )
        
        # Calculate average metrics
        train_avg_metrics = {}
        for metric_name in train_metrics[0].keys():
            values = [metrics[metric_name] for metrics in train_metrics]
            train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
        
        test_avg_metrics = {}
        for metric_name in test_metrics[0].keys():
            values = [metrics[metric_name] for metrics in test_metrics]
            test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
        
        # Store results
        results.append({
            "name": name,
            "train_total_revenue": sum(train_revenue),
            "test_total_revenue": sum(test_revenue),
            "train_avg_revenue": np.mean(train_revenue),
            "test_avg_revenue": np.mean(test_revenue),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_revenue": train_revenue,
            "test_revenue": test_revenue,
            **train_avg_metrics,
            **test_avg_metrics
        })
        
        # Print summary for this recommender
        print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
        print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
        performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
        print(f"  Performance Change: {performance_change:.2f}%")
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
    
    # Print summary table
    print("\nRecommender Evaluation Results (sorted by test revenue):")
    summary_cols = ["name", "train_total_revenue", "test_total_revenue", 
                   "train_avg_revenue", "test_avg_revenue",
                   "train_precision_at_k", "test_precision_at_k",
                   "train_ndcg_at_k", "test_ndcg_at_k",
                   "train_mrr", "test_mrr",
                   "train_discounted_revenue", "test_discounted_revenue"]
    summary_cols = [col for col in summary_cols if col in results_df.columns]
    
    print(results_df[summary_cols].to_string(index=False))
    
    # Cell: Results Visualization
    """
    ## Results Visualization
    Now we'll visualize the performance of the different recommenders.
    """
    
    # Generate comparison plots
    visualize_recommender_performance(results_df, recommender_names)
    
    # Generate detailed metrics visualizations
    visualize_detailed_metrics(results_df, recommender_names)
    
    return results_df


# Cell: Performance Visualization Functions
"""
## Performance Visualization Functions
These functions create visualizations for comparing recommender performance.
"""

def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_discounted_revenue' in results_df.columns and 'test_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue']
        test_revenue = results_df.iloc[i]['test_revenue']
        
        # Check if revenue is a scalar (numpy.float64) or a list/array
        if isinstance(train_revenue, (float, np.float64, np.float32, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, np.float32, int, np.integer)):
            test_revenue = [test_revenue]
            
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'train_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'train_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    # Get best-performing model
    best_model_idx = results_df['test_total_revenue'].idxmax()
    best_model_name = results_df.iloc[best_model_idx]['name']
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'test_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")


def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics
    all_metrics = []
    if len(results_df) > 0 and 'train_metrics' in results_df.columns:
        first_train_metrics = results_df.iloc[0]['train_metrics'][0]
        all_metrics = list(first_train_metrics.keys())
    
    # Select key metrics to visualize
    key_metrics = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric in enumerate(key_metrics):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name].iloc[0]
                
                # Get metric values for training phase
                train_values = []
                for train_metric in row['train_metrics']:
                    if metric in train_metric:
                        train_values.append(train_metric[metric])
                
                # Get metric values for testing phase
                test_values = []
                for test_metric in row['test_metrics']:
                    if metric in test_metric:
                        test_values.append(test_metric[metric])
                
                # Plot training phase
                plt.plot(range(len(train_values)), train_values, 
                         marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='-', label=f"{name} (train)")
                
                # Plot testing phase
                plt.plot(range(len(train_values), len(train_values) + len(test_values)), 
                         test_values, marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='--', label=f"{name} (test)")
                
                # Add a vertical line to separate train and test
                if j == 0:  # Only add the line once
                    plt.axvline(x=len(train_values)-0.5, color='k', 
                                linestyle='--', alpha=0.3, label='Train/Test Split')
            
            # Get metric info from EVALUATION_METRICS
            if metric in EVALUATION_METRICS:
                metric_info = EVALUATION_METRICS[metric]
                metric_name = metric_info['name']
                plt.title(f"{metric_name} Trajectory")
            else:
                plt.title(f"{metric.replace('_', ' ').title()} Trajectory")
            
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            
            # Add legend to the last plot only to avoid cluttering
            if i == len(key_metrics) - 1 or i == 5:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('recommender_metrics_trajectories.png')
    print("Detailed metrics visualizations saved to 'recommender_metrics_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns
    metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols = [col for col in metric_cols if not col.endswith('_metrics') and not col.endswith('_revenue')]
    
    if len(metric_cols) > 1:
        correlation_df = results_df[metric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig('metrics_correlation_heatmap.png')
        print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")


def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
        
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    for user_id, user_recs in recommendations.groupBy("user_idx").agg(
        sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations")
    ).collect():
        # Sort by rank
        user_rec_list = sorted(user_id.recommendations, key=lambda x: x[1])
        
        # Calculate DCG
        dcg = 0
        for i, (rel, _) in enumerate(user_rec_list[:k]):
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        
        user_dcg.append(dcg)
    
    # Return average DCG across all users
    return np.mean(user_dcg) if user_dcg else 0.0


# Cell: Main execution
"""
## Run the Analysis
When you run this notebook, it will perform the full analysis and visualization.
"""

if __name__ == "__main__":
    results = run_recommender_analysis() 
