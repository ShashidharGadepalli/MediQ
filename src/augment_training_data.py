import re
from preprocessing import read_reports_from_folder

def augment_report_abbreviations(text):
    """Replace full names with abbreviations."""
    replacements = {
        'Haemoglobin': 'Hb',
        'Total RBC': 'RBC',
        'Platelets': 'PLT',
        'Neutrophils': 'Neut',
        'Lymphocytes': 'Lymph',
        'Monocytes': 'Mono',
        'Eosinophils': 'Eos',
        'Haematocrit': 'Hct'
    }
    
    augmented = text
    for full, abbrev in replacements.items():
        augmented = augmented.replace(full, abbrev)
    
    return augmented

def augment_report_phrasing(text):
    """Change sentence structures."""
    replacements = {
        'came out to be': 'shows',
        'was measured at': 'recorded at',
        'Observed ': '',
        'Lab recorded ': '',
        'Marked as H': 'HIGH',
        'Marked as L': 'LOW'
    }
    
    augmented = text
    for old, new in replacements.items():
        augmented = augmented.replace(old, new)
    
    return augmented

def augment_report_flags(text):
    """Change flag formats."""
    augmented = text
    augmented = augmented.replace('Marked as H', 'elevated')
    augmented = augmented.replace('Marked as L', 'decreased')
    return augmented

def augment_report_spacing(text):
    """Add spacing variations."""
    augmented = text
    augmented = re.sub(r'(\d+\.?\d*)\s+(g/dL|mill/cmm|/uL|%)', r'\1\2', augmented)
    return augmented

def create_augmented_dataset():
    """Create augmented training dataset."""
    
    print("Loading original training reports...")
    original_reports = read_reports_from_folder("data/Train", limit=70)
    print(f"Loaded {len(original_reports)} original reports")
    
    augmented_data = []
    
    for filename, text in original_reports.items():
        augmented_data.append(text)
        
        var1 = augment_report_abbreviations(text)
        augmented_data.append(var1)
        
        var2 = augment_report_phrasing(text)
        augmented_data.append(var2)
        
        var3 = augment_report_flags(text)
        augmented_data.append(var3)
        
        var4 = augment_report_spacing(text)
        augmented_data.append(var4)
        
        var5 = augment_report_abbreviations(augment_report_phrasing(text))
        augmented_data.append(var5)
    
    print(f"Created {len(augmented_data)} training examples (original + augmented)")
    return augmented_data

if __name__ == "__main__":
    augmented = create_augmented_dataset()
    print(f"\nTotal training examples: {len(augmented)}")
    print("\nSample augmented report:")
    print("="*60)
    print(augmented[71][:500])
    print("...")