# Necessary import to generate the dataset
import os
import numpy as np
import pandas as pd
import datetime
import time
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # In case there is an error when importing logger and exception below
from exception import CustomException
from logger import logging

class DataGenerator:
    def __init__(self, 
                 n_customers=10000, 
                 n_terminals=1000000, 
                 nb_days=90, 
                 start_date="2018-04-01", 
                 radius=5):
        """
        Initialize the parameters to generate datas
        """
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.nb_days = nb_days
        self.start_date = start_date
        self.radius = radius
        
        # Configuration de la graine aléatoire pour reproductibilité
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

    def generate_customer_profiles(self):
        """
        Generate customer table
        """
        customer_profiles = []
        
        for customer_id in range(self.n_customers):
            profile = {
                'CUSTOMER_ID': customer_id,
                'x_customer_id': np.random.uniform(0, 100),
                'y_customer_id': np.random.uniform(0, 100),
                'mean_amount': np.random.uniform(5, 100),
                'std_amount': np.random.uniform(2.5, 50),
                'mean_nb_tx_per_day': np.random.uniform(0, 4)
            }
            customer_profiles.append(profile)
        
        return pd.DataFrame(customer_profiles)

    def generate_terminal_profiles(self):
        """
        Generate termninal table
        """
        terminal_profiles = []
        
        for terminal_id in range(self.n_terminals):
            profile = {
                'TERMINAL_ID': terminal_id,
                'x_terminal_id': np.random.uniform(0, 100),
                'y_terminal_id': np.random.uniform(0, 100)
            }
            terminal_profiles.append(profile)
        
        return pd.DataFrame(terminal_profiles)

    def get_available_terminals(self, customer_profiles, terminal_profiles):
        """
        Calculate all available terminal for one customer
        """
        x_y_terminals = terminal_profiles[['x_terminal_id', 'y_terminal_id']].values.astype(float)
        
        def find_terminals_in_radius(customer):
            x_y_customer = customer[['x_customer_id', 'y_customer_id']].values.astype(float)
            dist_x_y = np.sqrt(np.sum(np.square(x_y_customer - x_y_terminals), axis=1))
            return list(np.where(dist_x_y < self.radius)[0])
        
        customer_profiles['available_terminals'] = customer_profiles.apply(find_terminals_in_radius, axis=1)
        customer_profiles['nb_terminals'] = customer_profiles['available_terminals'].apply(len)
        
        return customer_profiles

    def generate_transactions(self, customer_profiles, terminal_profiles):
        """
        Generate transaction table
        """
        def generate_customer_transactions(customer_profile):
            customer_transactions = []
            
            for day in range(self.nb_days):
                nb_tx = np.random.poisson(customer_profile['mean_nb_tx_per_day'])
                
                if nb_tx > 0:
                    for _ in range(nb_tx):
                        time_tx = int(np.random.normal(86400/2, 20000))
                        
                        if 0 < time_tx < 86400:
                            amount = np.random.normal(
                                customer_profile['mean_amount'], 
                                customer_profile['std_amount']
                            )
                            
                            amount = max(0, amount)  # Évite les montants négatifs
                            amount = round(amount, 2)
                            
                            if customer_profile['nb_terminals'] > 0:
                                terminal_id = random.choice(
                                    customer_profile['available_terminals']
                                )
                                
                                transaction = {
                                    'TX_TIME_SECONDS': time_tx + day * 86400,
                                    'TX_TIME_DAYS': day,
                                    'CUSTOMER_ID': customer_profile['CUSTOMER_ID'],
                                    'TERMINAL_ID': terminal_id,
                                    'TX_AMOUNT': amount
                                }
                                customer_transactions.append(transaction)
            
            return pd.DataFrame(customer_transactions)

        transactions_df = customer_profiles.apply(
            generate_customer_transactions, axis=1
        )
        transactions_df = pd.concat(transactions_df.tolist())
        
        # Formatage des transactions
        transactions_df['TX_DATETIME'] = pd.to_datetime(
            transactions_df["TX_TIME_SECONDS"], 
            unit='s', 
            origin=self.start_date
        )
        
        transactions_df = transactions_df.sort_values('TX_DATETIME')
        transactions_df.reset_index(drop=True, inplace=True)
        transactions_df['TRANSACTION_ID'] = transactions_df.index
        
        return transactions_df
    
    def add_frauds(self, customer_profiles_table, terminal_profiles_table, transactions_df):
        '''
        This function is adding fraud case to the dataset
        '''
    
        # By default, all transactions are genuine
        transactions_df['TX_FRAUD']=0
        transactions_df['TX_FRAUD_SCENARIO']=0
        
        # Scenario 1
        transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
        transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
        
        # Scenario 2
        for day in range(transactions_df.TX_TIME_DAYS.max()):
            
            compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
            
            compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                        (transactions_df.TX_TIME_DAYS<day+28) & 
                                                        (transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                                
            transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
            transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO']=2
        
        # Scenario 3
        for day in range(transactions_df.TX_TIME_DAYS.max()):
            
            compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
            
            compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                        (transactions_df.TX_TIME_DAYS<day+14) & 
                                                        (transactions_df.CUSTOMER_ID.isin(compromised_customers))]
            
            nb_compromised_transactions=len(compromised_transactions)
            
            
            random.seed(day)
            index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/3))
            
            transactions_df.loc[index_fauds,'TX_AMOUNT']=transactions_df.loc[index_fauds,'TX_AMOUNT']*5
            transactions_df.loc[index_fauds,'TX_FRAUD']=1
            transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO']=3
        
        return transactions_df                 

    def generate_dataset(self):
        """
        Generate full dataset
        """
        start_time = time.time()
        
        logging.info("Start genrating the customer table")
        customer_profiles = self.generate_customer_profiles()
        terminal_profiles = self.generate_terminal_profiles()
        logging.info("Generation of the customer table finish")
        
        # Associating terminal with customer
        customer_profiles = self.get_available_terminals(
            customer_profiles, terminal_profiles
        )
        logging.info("Start generating the transaction table")
        transactions_df = self.generate_transactions(
            customer_profiles, terminal_profiles
        )
        logging.info("Generation of the transaction table finish")

        logging.info("Adding fraud to the dataset")
        transactions_df = self.add_frauds(
            customer_profiles, 
            terminal_profiles, 
            transactions_df
        )
        
        # os.makedirs('artifacts/data', exist_ok=True)
        # customer_profiles.to_csv('artifacts/data/customer_profiles.csv', index=False)
        # terminal_profiles.to_csv('artifacts/data/terminal_profiles.csv', index=False)
        # transactions_df.to_csv('artifacts/data/transactions.csv', index=False)

        DIR_OUTPUT = "artifacts/simulated-data-raw/"

        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)

        start_date = datetime.datetime.strptime("2018-04-01", "%Y-%m-%d")

        for day in range(transactions_df.TX_TIME_DAYS.max()+1):
            
            transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==day].sort_values('TX_TIME_SECONDS')
            
            date = start_date + datetime.timedelta(days=day)
            filename_output = date.strftime("%Y-%m-%d")+'.pkl'
            
            # Protocol=4 required for Google Colab
            transactions_day.to_pickle(DIR_OUTPUT+filename_output, protocol=4)
                
        logging.info(f"Generation of datas finish in : {time.time() - start_time:.2f} secondes")
        
        return customer_profiles, terminal_profiles, transactions_df

if __name__ == "__main__":
    try:
        generator = DataGenerator(
            n_customers=5000, 
            n_terminals=10000, 
            nb_days=183
        )
        generator.generate_dataset()
    except Exception as e:
        raise CustomException(e, sys)
