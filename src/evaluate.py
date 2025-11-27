import spacy
import json
import os
import re
from difflib import SequenceMatcher
from preprocessing import read_reports_from_folder
from extraction import extract_all
from dotenv import load_dotenv

load_dotenv()

def fuzzy_match(str1, str2, threshold=0.8):
    """Check if two strings are similar using fuzzy matching."""
    if str1 is None or str2 is None:
        return str1 == str2
    return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio() >= threshold

def evaluate_patient_info(predicted, ground_truth):
    """Evaluate patient information extraction."""
    correct = 0
    total = 0
    
    for key in ground_truth:
        total += 1
        if key in predicted:
            if fuzzy_match(predicted[key], ground_truth[key]):
                correct += 1
    
    return correct, total

def evaluate_labs(predicted_labs, ground_truth_labs):
    """Evaluate lab results with fuzzy matching."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    gt_dict = {lab['test']: lab for lab in ground_truth_labs}
    pred_dict = {lab['test']: lab for lab in predicted_labs}
    
    for test_name in gt_dict:
        gt_lab = gt_dict[test_name]
        
        matched_pred = None
        for pred_name in pred_dict:
            if fuzzy_match(test_name, pred_name, threshold=0.85):
                matched_pred = pred_dict[pred_name]
                break
        
        if matched_pred:
            if 'value' in gt_lab:
                if 'value' in matched_pred and abs(matched_pred['value'] - gt_lab['value']) < 0.1:
                    true_positives += 1
                else:
                    false_negatives += 1
            
            if 'unit' in gt_lab:
                if 'unit' in matched_pred and fuzzy_match(matched_pred['unit'], gt_lab['unit']):
                    true_positives += 1
                else:
                    false_negatives += 1
            
            if 'flag' in gt_lab:
                if 'flag' in matched_pred and matched_pred['flag'] == gt_lab['flag']:
                    true_positives += 1
                else:
                    false_negatives += 1
        else:
            fields_count = sum(1 for k in ['value', 'unit', 'flag'] if k in gt_lab)
            false_negatives += fields_count
    
    for pred_name in pred_dict:
        if not any(fuzzy_match(pred_name, gt_name, threshold=0.85) for gt_name in gt_dict):
            fields_count = sum(1 for k in ['value', 'unit', 'flag'] if k in pred_dict[pred_name])
            false_positives += fields_count
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_positives, false_positives, false_negatives

def evaluate_structure(extracted_data):
    """Check completeness and structural consistency."""
    issues = []
    score = 100
    
    required_fields = ['patient', 'labs', 'diagnosis', 'medications']
    for field in required_fields:
        if field not in extracted_data:
            issues.append(f"Missing required field: {field}")
            score -= 25
    
    if 'patient' in extracted_data:
        patient_required = ['name', 'age', 'gender', 'id']
        for field in patient_required:
            if field not in extracted_data['patient']:
                issues.append(f"Missing patient field: {field}")
                score -= 5
    
    if 'labs' in extracted_data:
        if not isinstance(extracted_data['labs'], list):
            issues.append("Labs should be a list")
            score -= 10
        else:
            for i, lab in enumerate(extracted_data['labs']):
                if 'test' not in lab:
                    issues.append(f"Lab {i} missing test name")
                    score -= 2
                if 'value' not in lab and 'unit' not in lab:
                    issues.append(f"Lab {i} ({lab.get('test', 'unknown')}) has neither value nor unit")
                    score -= 2
    
    if 'diagnosis' in extracted_data:
        if not isinstance(extracted_data['diagnosis'], list):
            issues.append("Diagnosis should be a list")
            score -= 5
    
    if 'medications' in extracted_data:
        if not isinstance(extracted_data['medications'], list):
            issues.append("Medications should be a list")
            score -= 5
    
    return max(0, score), issues

def llm_evaluation(original_text, extracted_data, api_key=None):
    """Use LLM to semantically evaluate extraction quality."""
    
    if not api_key:
        return None, "LLM evaluation skipped (no API key provided)"
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are evaluating a medical NLP extraction system.

ORIGINAL REPORT:
{original_text}

EXTRACTED STRUCTURED DATA:
{json.dumps(extracted_data, indent=2)}

Evaluate how accurately the structured data captures the original report.

Provide:
1. Semantic Accuracy Score (0-100): How well does the JSON represent the report?
2. What was extracted correctly?
3. What was missed or extracted incorrectly?
4. Overall assessment (2-3 sentences)

Format your response as:
SCORE: [number]
CORRECT: [brief list]
ISSUES: [brief list]
ASSESSMENT: [2-3 sentences]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, f"LLM evaluation failed: {str(e)}"

def load_ground_truth(filename):
    """Load ground truth JSON."""
    match = re.search(r'(\d+)', filename)
    if match:
        report_num = match.group(1)
        gt_file = f"data/Ground-TruthJsons/test{report_num}.json"
    else:
        gt_file = f"data/Ground-TruthJsons/{filename.replace('.txt', '.json')}"
    
    with open(gt_file, 'r') as f:
        return json.load(f)

def evaluate_test_set(llm_api_key=None):
    """Run complete evaluation on test set."""
    
    nlp = spacy.load("medical_ner_model_v2")
    test_reports = read_reports_from_folder("data/Test")
    
    all_results = []
    llm_summary_scores = []
    
    for filename, text in test_reports.items():
        predicted = extract_all(text, nlp)
        ground_truth = load_ground_truth(filename)
        
        precision, recall, f1, tp, fp, fn = evaluate_labs(predicted['labs'], ground_truth['labs'])
        struct_score, struct_issues = evaluate_structure(predicted)
        
        llm_result = None
        llm_error = None
        if llm_api_key:
            llm_result, llm_error = llm_evaluation(text, predicted, llm_api_key)
        else:
            llm_error = "No API key provided"
        
        all_results.append({
            'filename': filename,
            'entity_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            },
            'structural_score': struct_score,
            'structural_issues': struct_issues,
            'llm_evaluation': llm_result if llm_result else llm_error
        })
        
        if llm_result:
            try:
                score_line = [line for line in llm_result.split('\n') if 'SCORE:' in line][0]
                score = int(score_line.split(':')[1].strip())
                llm_summary_scores.append(score)
            except:
                pass
    
    avg_precision = sum(r['entity_metrics']['precision'] for r in all_results) / len(all_results)
    avg_recall = sum(r['entity_metrics']['recall'] for r in all_results) / len(all_results)
    avg_f1 = sum(r['entity_metrics']['f1'] for r in all_results) / len(all_results)
    avg_struct = sum(r['structural_score'] for r in all_results) / len(all_results)
    avg_llm = sum(llm_summary_scores) / len(llm_summary_scores) if llm_summary_scores else None
    
    results = {
        'individual_results': all_results,
        'summary': {
            'entity_level': {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1
            },
            'structural': {
                'avg_score': avg_struct
            },
            'llm_based': {
                'avg_semantic_score': avg_llm,
                'num_evaluated': len(llm_summary_scores)
            }
        }
    }
    
    output_file = "output/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    load_dotenv()
    llm_key = os.getenv("OPENAI_API_KEY")
    evaluate_test_set(llm_api_key=llm_key)