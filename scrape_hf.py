from huggingface_hub import HfApi
import pandas as pd
import time

def categorize_domain(tags, model_name, task):
    text_to_search = f"{' '.join(tags if tags else [])} {model_name} {task}".lower()
    
    domain_keywords = {
        "Healthcare & Medicine": ["health", "medic", "clinic", "bio", "covid", "disease", "hospital", "doctor", "cancer", "xray"],
        "Finance & Economy": ["finance", "bank", "stock", "trade", "econom", "rupee", "business", "market"],
        "Law & Legal": ["law", "legal", "court", "justice", "constitution", "act", "supreme"],
        "Engineering & Tech": ["code", "programming", "software", "engineering", "math", "tech", "algorithm", "python"],
        "Education & Research": ["education", "school", "college", "university", "research", "student", "study"],
        "Language & Linguistics": ["translation", "grammar", "dictionary", "asr", "tts", "pos-tagging", "ner", "speech", "text-to-speech"],
        "Agriculture & Environment": ["agri", "farm", "crop", "plant", "soil", "weather", "climate", "forest"]
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in text_to_search for keyword in keywords):
            return domain
            
    return "Uncategorized"

def main():
    print("Connecting to Hugging Face API...")
    api = HfApi()

    search_terms = ["nepali", "nepal", "nepalese"]
    all_models = []
    seen_models = set()

    print("Fetching models... This might take a few seconds.")
    
    for term in search_terms:
        print(f"Searching for keyword: '{term}'...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                models = api.list_models(search=term)
                
                for model in models:
                    if model.modelId in seen_models:
                        continue
                    seen_models.add(model.modelId)
                    
                    tags = model.tags if model.tags else []
                    model_name = model.modelId
                    task = model.pipeline_tag if model.pipeline_tag else ""
                    
                    domain = categorize_domain(tags, model_name, task)
                    
                    model_info = {
                        "Domain": domain,                   # <--- NEW DOMAIN COLUMN
                        "Model Name": model_name,
                        "Author": model.author if model.author else "Unknown",
                        "Task": task if task else "None",
                        "Downloads": model.downloads,
                        "Likes": model.likes,
                        "Tags": ", ".join(tags) if tags else "None",
                        "URL": f"https://huggingface.co/{model_name}"
                    }
                    all_models.append(model_info)
                
                break 
                
            except Exception as e:
                print(f"⚠️ Connection interrupted. (Attempt {attempt + 1} of {max_retries}). Retrying in 5 seconds...")
                time.sleep(5)
                
        time.sleep(2) 

    print(f"\nSuccessfully found {len(all_models)} models!")

    if len(all_models) > 0:
        df = pd.DataFrame(all_models)
        
        df = df.sort_values(by=["Domain", "Downloads"], ascending=[True, False])
        
        excel_filename = "Categorized_Nepalese_HF_Models.xlsx"
        df.to_excel(excel_filename, index=False)
        
        print(f"Done! All data saved to '{excel_filename}'.\n")
        
        print("📊 --- DOMAIN SUMMARY --- 📊")
        print(df['Domain'].value_counts().to_string())
        
    else:
        print("No models were found.")

if __name__ == "__main__":
    main()