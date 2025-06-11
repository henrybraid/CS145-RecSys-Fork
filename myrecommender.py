# Cell: Define custom recommender template
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import sklearn 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sim4rec.utils import pandas_to_spark

import xgboost as xgb
from xgboost import XGBClassifier, callback

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import torch_geometric.nn as geom_nn
from torch_geometric.data import Data


"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""

class GradientBoost:
    """
    Template class for implementing a custom recommender.
    
    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """
    
    def __init__(self, seed=None):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        # Add your initialization logic here
        self.seed = seed
        self.categorical_cols = None
        self.numerical_cols = None
        self.input_cols = None
        self.pipeline = None
        self.model = None
        self.best_params = None
        self.encoder = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
        self.scalar = StandardScaler()

    def _create_features(self, features):
        #average category price
        if 'i_category' in features.columns and 'i_price' in features.columns:
            features['avg_category_price'] = features.groupby('i_category')['i_price'].transform('mean')
        
        #get the average price spent by user
        if 'user_idx' in features.columns and 'i_price' in features.columns:
            features['user_avg_price'] = features.groupby('user_idx')['i_price'].transform('mean')

        #get the price of the item compared to the average amount spent by the users
        if 'user_avg_price' in features.columns and 'i_price' in features.columns:
            features['price_vs_user_mean'] = features['i_price'] - features['user_avg_price']
        
        return features

    
    def _setup_df(self, log, user_features = None, item_features = None):
        #add 'u_' prefix to the user features, helps with clarity
        user_features = user_features.select(
            [sf.col('user_idx')] + 
            [sf.col(c).alias(f'u_{c}') for c in user_features.columns if c != 'user_idx']
        )

        #add 'i_' prefix to the item features, helps with clarity
        item_features = item_features.select(
            [sf.col('item_idx')] + 
            [sf.col(c).alias(f'i_{c}') for c in item_features.columns if c != 'item_idx']
        )

        pd_log = (
            log.alias('l')
                .join(user_features.alias('u'), on='user_idx', how = 'inner')
                .join(item_features.alias('i'), on='item_idx', how = 'inner')
                .toPandas()
        )

        return pd_log, user_features, item_features
    
    def _preprocess_features(self, features):
        self.categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.input_cols = self.categorical_cols + self.numerical_cols
        

        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', self.encoder)
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', self.scalar)
        ])

        self.pipeline = ColumnTransformer(
            transformers = [
                ('cat', cat_pipeline, self.categorical_cols),
                ('num', num_pipeline, self.numerical_cols)
            ]
        )

        features = features.reindex(columns=self.input_cols)
        features_transformed = self.pipeline.fit_transform(features)

        return features_transformed

    def _get_best_model(self, X,y):
        #Best params so far: 31% increase, commented on each side
        param_grid = {
            "n_estimators": [10, 25, 100], #25
            "learning_rate": [0.001, 0.01], #0.01
            "max_depth": [2, 4], #4
            "min_child_weight": [4, 5, 6], #5
            'reg_lambda':[1.0], #when using other regularization, performance decreased
        }
        base_model = XGBClassifier(
                    objective="binary:logistic", 
                    booster='gbtree',
                    random_state = self.seed, 
                    tree_method = 'hist',
                    eval_metric='logloss',
                    n_jobs = 4)
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv = 3, 
            scoring = 'neg_mean_squared_error',
            n_jobs = 1)
        grid_search.fit(X, y,verbose=False)

        self.best_params = grid_search.best_params_

        return grid_search.best_estimator_

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        # Implement your training logic here
        # For example:
        #  1. Extract relevant features from user_features and item_features
        #  2. Learn user preferences from the log
        #  3. Build item similarity matrices or latent factor models
        #  4. Store learned parameters for later prediction
        if user_features and item_features:
            pd_log, user_features, item_features = self._setup_df(log, user_features, item_features)
            pd_log = self._create_features(pd_log)
            features = pd_log.drop(columns=['user_idx', 'item_idx', 'relevance'])

            X = self._preprocess_features(features)
            y = pd_log['relevance'].values

            if self.model is None:
                #Perform grid search to get the best model on first fit. Then, use this model for the rest of training.
                self.model = self._get_best_model(X,y)
                print(f'\nBest parameters: {self.best_params}\n')

            else:
                #Apply early stopping to the training iterations after determining the best model
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=self.seed)
                
                self.model = XGBClassifier(
                            **self.best_params,
                            random_state=self.seed,
                            booster='gbtree',
                            tree_method='hist',
                            eval_metric='logloss',
                            early_stopping_rounds = 5, #Does not even happen, because the 25 estimators is already performing well
                            n_jobs=4)
                self.model.fit(X_train,y_train, 
                               eval_set = [(X_test, y_test)],
                               verbose=False)
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        # Implement your recommendation logic here
        # For example:
        #  1. Extract relevant features for prediction
        #  2. Calculate relevance scores for each user-item pair
        #  3. Rank items by relevance and select top-k
        #  4. Return a dataframe with columns: user_idx, item_idx, relevance
        candidate_df = users.crossJoin(items)

        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx").distinct()
            candidate_df = candidate_df.join(seen, ["user_idx", "item_idx"], "left_anti")

        candidate_pd, _, _ = self._setup_df(candidate_df, user_features, item_features)

        candidate_pd = self._create_features(candidate_pd)

        meta_pd = candidate_pd[["user_idx", "item_idx"]].copy()

        features = candidate_pd.drop(
                        columns=[c for c in ["__iter", "relevance"] if c in candidate_pd.columns],
                        errors="ignore"
                )
    
        features = features.reindex(columns=self.input_cols, fill_value=np.nan)

        X = self.pipeline.transform(features)

        meta_pd["relevance"] = self.model.predict_proba(X)[:, 1]

        #Rank and take top k
        topk_pd = (
            meta_pd.sort_values(["user_idx", "relevance"], ascending=[True, False])
                .groupby("user_idx")
                .head(k)
            )
    
        return pandas_to_spark(topk_pd[["user_idx", "item_idx", "relevance"]])


class RevenueRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, num_layers = 1, dropout= 0.0, nonlinearity='tanh'):
        super().__init__()

        self.input_size = input_dim
        self.model = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout=dropout if num_layers>1 else 0.0,
            nonlinearity=nonlinearity,
            batch_first = True
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x_packed):

        # x_packeed is a PackedSequence of shape (B, L, D)
        packed_out, _ = self.model(x_packed)

        # unpack back to (B, L, hidden_dim)
        out, lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # apply linear layer at each time-step → (B, L, 1)
        rev = self.out(out)
        return rev.squeeze(-1)



class RnnRecommender():
    
    def __init__(self, seed, hidden_dim=128, num_layers=1, dropout=0.0, lr=1e-3):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.encoder = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
        self.scalar = StandardScaler()

    def _create_features(self, features):
        #Use the row ordering as the timestamping
        features = features.reset_index(drop=True)
        features['timestamp'] = features.index

        #average category price
        if 'i_category' in features.columns and 'i_price' in features.columns:
            features['avg_category_price'] = features.groupby('i_category')['i_price'].transform('mean')
        
        #get the average price spent by user
        if 'user_idx' in features.columns and 'i_price' in features.columns:
            features['user_avg_price'] = features.groupby('user_idx')['i_price'].transform('mean')

        #get the price of the item compared to the average amount spent by the users
        if 'user_avg_price' in features.columns and 'i_price' in features.columns:
            features['price_vs_user_mean'] = features['i_price'] - features['user_avg_price']

        return features


    def _setup_df(self, log, user_features = None, item_features = None):
        #add 'u_' prefix to the user features, helps with clarity
        user_features = user_features.select(
            [sf.col('user_idx')] + 
            [sf.col(c).alias(f'u_{c}') for c in user_features.columns if c != 'user_idx']
        )

        #add 'i_' prefix to the item features, helps with clarity
        item_features = item_features.select(
            [sf.col('item_idx')] + 
            [sf.col(c).alias(f'i_{c}') for c in item_features.columns if c != 'item_idx']
        )

        pd_log = (
            log.alias('l')
                .join(user_features.alias('u'), on='user_idx', how = 'inner')
                .join(item_features.alias('i'), on='item_idx', how = 'inner')
                .toPandas()
        )

        return pd_log, user_features, item_features

    def _preprocess_features(self, features):
        self.categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.input_cols = self.categorical_cols + self.numerical_cols
        

        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', self.encoder)
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', self.scalar)
        ])

        self.pipeline = ColumnTransformer(
            transformers = [
                ('cat', cat_pipeline, self.categorical_cols),
                ('num', num_pipeline, self.numerical_cols)
            ]
        )

        features = features.reindex(columns=self.input_cols)
        features_transformed = self.pipeline.fit_transform(features)

        return features_transformed
    
    def _build_sequences(self, pd_log, X_np, y_np):
        X_seq, y_seq, lengths = [], [], []
        
        for uid, grp in pd_log.groupby('user_idx', sort = False):
            idx = grp.sort_values('timestamp').index
            features = torch.tensor(X_np[idx], dtype=torch.float32)
            targets = torch.tensor(y_np[idx],dtype=torch.float32)
            X_seq.append(features)
            y_seq.append(targets)
            lengths.append(len(idx))
        
        X_pad = nn.utils.rnn.pad_sequence(X_seq, batch_first = True)
        y_pad = nn.utils.rnn.pad_sequence(y_seq, batch_first = True)
        lengths = torch.tensor(lengths)
        
        return X_pad, y_pad, lengths
        

    def _init_rnn(self, input_dim):
        self.model = RevenueRNN(
            input_dim=input_dim, 
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            dropout = self.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def fit(self, log, user_features = None, item_features = None):
        """
         Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        
        """

        pd_log, user_features, item_features = self._setup_df(log, user_features, item_features)
        pd_log = self._create_features(pd_log)
        features = pd_log.drop(columns=['user_idx', 'item_idx', 'relevance'])

        #Send the features through the data processing pipeline
        features_transformed = self._preprocess_features(features)

        X_np = features_transformed.toarray() if hasattr(features_transformed, "toarray") else features_transformed
        y_np = pd_log['relevance'].values

        #Make the data sequential and ordered by row index
        X, y, lengths = self._build_sequences(pd_log, X_np, y_np)
        
        #Pack data for the RNN
        X_packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)

        current_dim = X.shape[-1]
        if (self.model is None) or (self.model.input_size != current_dim):
            self._init_rnn(current_dim)
        
        self.model.train()
        self.optimizer.zero_grad()

        # forward
        preds = self.model(X_packed)  # (batch, seq_len)

        # compute loss only over the valid time-steps
        # mask out padded positions
        mask = (torch.arange(preds.size(1))[None, :].to(self.device)< lengths[:, None])
        loss = self.criterion(preds[mask], y[mask])

        # backward + step
        loss.backward()
        self.optimizer.step()

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        log = log.toPandas()
        users = users.toPandas()
        items = items.toPandas()
        user_features = user_features.toPandas()
        item_features = item_features.toPandas()
        
        price_map = items.set_index("item_idx")["price"   ].to_dict() if "price"    in items else {}
        category_map = items.set_index("item_idx")["category"].to_dict() if "category" in items else {}

        # Group past interactions once
        hist_by_user = log.groupby("user_idx")
        
        self.model.eval()

        recommendations = []

        for uid in users['user_idx'].unique():
            #Build the user's history:
            if uid in hist_by_user.groups:
                past = hist_by_user.get_group(uid).copy()
                past = self._create_features(past)   # adds timestamp & aggregates
                hist_items = past["item_idx"].tolist()
            else:
                past = pd.DataFrame(columns=log.columns)
                past = self._create_features(past)
                hist_items = []

            cand_items = items["item_idx"].tolist()
            if filter_seen_items:
                cand_items = [it for it in cand_items if it not in hist_items]

            scores = []
            for it in cand_items:
                row = {
                    "user_idx": uid,
                    "item_idx": it,
                    **{c: user_features.loc[user_features["user_idx"] == uid, c].iloc[0]
                    for c in user_features.columns if c != "user_idx"},
                    **{c: item_features.loc[item_features["item_idx"] == it, c].iloc[0]
                    for c in item_features.columns if c != "item_idx"},
                }
                row["timestamp"] = len(hist_items)
                next_df = pd.DataFrame([row])
                next_df = self._create_features(next_df)

                # Transform history + candidate together to ensure equal width
                seq_df = pd.concat([past, next_df], ignore_index=True)
                seq_df = seq_df.reindex(columns=self.input_cols)
                X_seq  = self.pipeline.transform(seq_df)

                X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, F)

                # pack exactly as in training
                lengths = torch.tensor([X_seq.shape[0]], dtype=torch.long)
                packed  = nn.utils.rnn.pack_padded_sequence(
                    X_tensor, lengths, batch_first=True, enforce_sorted=False
                )

                with torch.no_grad():
                    y_pred_seq = self.model(packed)
                    score = y_pred_seq[0, -1].item() # last timestep

                #expected revenue
                price = price_map.get(it, 1.0)
                expected_rev = score * price
                scores.append((it, expected_rev))
            top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
            for rank, (it, sc) in enumerate(top_k, 1):
                recommendations.append({
                    "user_idx": uid,
                    "item_idx": it,
                    "relevance": sc,
                    "rank": rank
                })
        rec_pd = pd.DataFrame(recommendations)
        rec_spark = spark.createDataFrame(rec_pd)
        return rec_spark
    


class RevenueGCN(nn.Module):
    def __init__(self, user_dim, item_dim, common_dim = 64, hidden_dim = 64, output_dim = 32):
        super(RevenueGCN, self).__init__()

        #Projection layers (because user and item have different dimensions)
        self.user_proj = nn.Linear(user_dim, common_dim)
        self.item_proj = nn.Linear(item_dim, common_dim)

        #GCN Layers:
        self.conv1 = geom_nn.GCNConv(in_channels=common_dim, out_channels = hidden_dim)
        self.conv2 = geom_nn.GCNConv(in_channels = hidden_dim, out_channels = output_dim)
    
    def forward(self, user_tensor, item_tensor, edge_index, edge_weight = None):
        #project features into embeddings:
        user_embedding = self.user_proj(user_tensor)
        item_embedding = self.item_proj(item_tensor)

        x = torch.cat([user_embedding, item_embedding], dim=0)

        #GCN layers:
        x = self.conv1(x, edge_index, edge_weight)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x




class GCNRecommender:
    def __init__(self, seed):
        self.seed = seed

        self.model = None

    def _create_features(self, features):
        #average category price
        if 'i_category' in features.columns and 'i_price' in features.columns:
            features['avg_category_price'] = features.groupby('i_category')['i_price'].transform('mean')
        
        #get the average price spent by user
        if 'user_idx' in features.columns and 'i_price' in features.columns:
            features['user_avg_price'] = features.groupby('user_idx')['i_price'].transform('mean')

        #get the price of the item compared to the average amount spent by the users
        if 'user_avg_price' in features.columns and 'i_price' in features.columns:
            features['price_vs_user_mean'] = features['i_price'] - features['user_avg_price']

        return features

    def _setup_df(self, log, user_features = None, item_features = None):
        pd_log = (
            log.alias('l')
                .join(user_features.alias('u'), on='user_idx', how = 'inner')
                .join(item_features.alias('i'), on='item_idx', how = 'inner')
                .toPandas()
        )

        return pd_log, user_features, item_features
    
    def _preprocess_features(self, features):
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        input_cols = categorical_cols + numerical_cols
        
        encoder = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
        scalar = StandardScaler()

        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', encoder)
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', scalar)
        ])

        pipeline = ColumnTransformer(
            transformers = [
                ('cat', cat_pipeline, categorical_cols),
                ('num', num_pipeline, numerical_cols)
            ]
        )

        features = features.reindex(columns=input_cols)
        features_transformed = pipeline.fit_transform(features)

        return features_transformed
    
    def _get_graph_pieces(self, log, user_features = None, item_features = None):
        #Get amount of users and items
        num_users = user_features['user_idx'].nunique()
        num_items = item_features['item_idx'].nunique()

        user_features = user_features.sort_values('user_idx')
        item_features = item_features.sort_values('item_idx')

        # Offset item_idx so item nodes start after user nodes
        item_offset = num_users
        log['item_idx'] = log['item_idx'] + item_offset

        #build edge index
        edge_index = torch.tensor([
            log['user_idx'].values,
            log['item_idx'].values
            ], 
            dtype = torch.long)
        
        #temporary, might want to make more complex
        edge_weight = torch.tensor(log['relevance'].values, dtype = torch.float)

        #Preprocess features
        user_features_transformed = self._preprocess_features(user_features)
        item_features_transformed = self._preprocess_features(item_features)

        user_tensor = torch.tensor(user_features_transformed, dtype = torch.float)
        item_tensor = torch.tensor(item_features_transformed, dtype = torch.float)
        
        return user_tensor, item_tensor, edge_index, edge_weight


    def fit(self, log, user_features = None, item_features = None):
        #Make them into pandas dataframe
        pd_log = log.toPandas()
        if hasattr(user_features, "toPandas"):
            user_features = user_features.toPandas()
        if hasattr(item_features, "toPandas"):
            item_features = item_features.toPandas()
        user_tensor, item_tensor, edge_index, edge_weight = self._get_graph_pieces(pd_log, user_features, item_features)
        print(f"Got user pieces")
        #get necessary dimensions
        user_dim = user_tensor.shape[1]
        item_dim = item_tensor.shape[1]
        
        if self.model is None:
            self.model = RevenueGCN(user_dim=user_dim, item_dim=item_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.criterion = torch.nn.MSELoss()

        #Train the model:
        self.model.train()
        self.optimizer.zero_grad()

        # Get node embeddings
        node_embeddings = self.model(user_tensor, item_tensor, edge_index, edge_weight)

        # Split embeddings
        user_embs = node_embeddings[:user_tensor.shape[0]]
        item_embs = node_embeddings[user_tensor.shape[0]:]

        # For each edge in edge_index[0] (user_idx) and edge_index[1] (item_idx)
        # compute dot product between user embedding and item embedding → predicted relevance
        user_edge_emb = user_embs[edge_index[0]]
        item_edge_emb = item_embs[edge_index[1] - user_tensor.shape[0]]  # shift back offset

        # Predicted relevance
        preds = torch.sum(user_edge_emb * item_edge_emb, dim=1)  # [num_edges]
        loss = self.criterion(preds, edge_weight)

        # backward + step
        loss.backward()
        self.optimizer.step()
        

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        pass