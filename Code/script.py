import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading data
interaction_df = pd.read_csv('Data/interactions.csv')
metadata_df = pd.read_csv('Data/metadata.csv')

basket = interaction_df.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0)

#groupby → groups by invoice and product
#sum() → adds up quantities
#unstack() → pivots StockCode into columns
#fillna(0) → fills missing combinations with 0

basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)

#Converts all numbers into just 0 or 1 (bought or not bought), because association rules don't care about quantity,
#just whether the item was present, converted to bool as fpgrowth algorithm works better with it

#ar part
min_support = 0.005
min_confidence = 0.5
min_lift = 3
frequent_itemsets = fpgrowth(basket, min_support, use_colnames=True) #running fp_growth
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence) #generating rules
rules = rules[rules['lift'] >= min_lift]

#content-based part
tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(metadata_df['Description'])
content_sim_matrix = cosine_similarity(vectors, vectors)
item_index_mapping = pd.Series(metadata_df.index, index=metadata_df['StockCode']).to_dict() #maps stockcode -> rowindex so we can find directly
reverse_index_mapping = {v: k for k, v in item_index_mapping.items()} #rowindex -> stockcode

def get_candidates(customer_id, df_transactions, ar_rules, similarity_matrix, top_k=5):
    
    #get the user's history
    user_history = set(df_transactions[df_transactions['Customer ID'] == customer_id]['StockCode'].unique())
    
    if not user_history:
        return {}, {}

    #generate AR Candidates
    ar_candidates = {}
    
    #filter rules where the antecedent is a subset of the user's history
    matching_rules = ar_rules[ar_rules['antecedents'].apply(lambda x: x.issubset(user_history))]
    
    for _, row in matching_rules.iterrows():
        for item in row['consequents']:
            if item not in user_history:
                
                ar_candidates[item] = max(ar_candidates.get(item, 0), row['confidence']) #store the highest confidence score for each candidate

    #generate Content-Based Candidates
    content_candidates = {}
    
    for item in user_history:
        if item in item_index_mapping:
            
            idx = item_index_mapping[item] 
            sim_scores = list(enumerate(similarity_matrix[idx])) #get similarity scores for this item against all others
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1] #sort and take top k similar items
            
            for sim_idx, score in sim_scores:
                
                candidate_item = reverse_index_mapping.get(sim_idx)
                
                if candidate_item and candidate_item not in user_history:
                    content_candidates[candidate_item] = max(content_candidates.get(candidate_item, 0), score)
                    
    return ar_candidates, content_candidates

def rank_hybrid_recommendations(ar_candidates, content_candidates, alpha=0.5, k=5):
    
    combined_items = set(ar_candidates.keys()).union(set(content_candidates.keys()))
    final_scores = []

    for item in combined_items:
        ar_score = ar_candidates.get(item, 0)
        content_score = content_candidates.get(item, 0)
        score = (alpha * ar_score) + ((1 - alpha) * content_score) #weighted hybrid formula
        final_scores.append({
            'StockCode': item,
            'HybridScore': score,
            'AR_Score': ar_score,
            'Content_Score': content_score
        })

    
    ranked_list = sorted(final_scores, key=lambda x: x['HybridScore'], reverse=True) #sort by hybrid_score descending
    return pd.DataFrame(ranked_list).head(k)

def main():

    while True:
    
        #get user input
        customer_id = int(input("Enter Customer ID: "))
        alpha = float(input("Enter alpha (0.0-1.0): "))

        if not 0.0 <= alpha <= 1.0:
            print("Alpha must be between 0.0 and 1.0")
            return
        
        top_k = int(input("Enter number of recommendations (default 5): ") or 5)
        ar_cands, ct_cands = get_candidates(customer_id, interaction_df, rules, content_sim_matrix, top_k) #recommendations

        if not ar_cands and not ct_cands:
            print(f"Customer {customer_id} not found or has no purchase history.")
            return
        
        recs = rank_hybrid_recommendations(ar_cands, ct_cands, alpha=alpha, k=top_k)
        recs = recs.merge(metadata_df[['StockCode', 'Description']], on='StockCode', how='left') #merge with descriptions
        
        print(f"\nTop {top_k} Recommendations for Customer {customer_id} (α={alpha}):")
        print(recs[['StockCode', 'Description', 'HybridScore', 'AR_Score', 'Content_Score']])

if __name__ == "__main__":
    main()