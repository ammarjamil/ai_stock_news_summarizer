
import json
import re
from typing import Dict, Any
from langchain.schema import BaseOutputParser


class JSONOutputParser(BaseOutputParser):
    """Custom parser to extract JSON from LLM response"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        print(f"Raw LLM Response (first 500 chars): {text[:500]}")
        print(f"Raw LLM Response (last 200 chars): {text[-200:]}")
        
        try:
            # Method 1: Try to find JSON content between ```json and ``` markers
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
                print("Found JSON in code block")
            else:
                # Method 2: Try to find JSON content between ``` markers (without json label)
                json_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    print("Found JSON in generic code block")
                else:
                    # Method 3: Find JSON by looking for balanced braces
                    json_str = self._extract_json_by_braces(text)
                    if not json_str:
                        print("No JSON found in braces, trying full text cleaning")
                        json_str = self._clean_and_extract_json(text)
            
            # Clean the JSON string
            json_str = json_str.strip()
            if not json_str:
                raise ValueError("No valid JSON content found")
            
            print(f"Extracted JSON string (first 300 chars): {json_str[:300]}")
            
            # Try to fix common issues before parsing
            json_str = self._fix_common_json_issues(json_str)
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            print("Successfully parsed JSON")
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Problematic JSON string: {json_str[:500] if 'json_str' in locals() else 'Not extracted'}")
            
            # Try aggressive fixing
            if 'json_str' in locals():
                fixed_json = self._aggressive_json_fix(json_str)
                try:
                    return json.loads(fixed_json)
                except Exception as fix_error:
                    print(f"Even fixed JSON failed: {fix_error}")
            
            raise ValueError(f"Invalid JSON in response: {e}")
        except Exception as e:
            print(f"General parsing error: {e}")
            raise ValueError(f"Error parsing response: {e}")
    
    def _extract_json_by_braces(self, text: str) -> str:
        """Extract JSON by finding balanced braces"""
        try:
            start_idx = text.find('{')
            if start_idx == -1:
                return ""
            
            brace_count = 0
            end_idx = start_idx
            
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count == 0:
                return text[start_idx:end_idx + 1]
            return ""
        except Exception as e:
            print(f"Error in brace extraction: {e}")
            return ""
    
    def _clean_and_extract_json(self, text: str) -> str:
        """Clean text and try to extract JSON-like content"""
        try:
            # Remove everything before first { and after last }
            start_brace = text.find('{')
            end_brace = text.rfind('}')
            
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                return text[start_brace:end_brace + 1]
            
            return ""
        except Exception as e:
            print(f"Error in text cleaning: {e}")
            return ""
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        try:
            # Remove any non-printable characters
            json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
            
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix single quotes to double quotes for strings
            json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
            
            # Fix unquoted keys (common issue)
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # Fix boolean/null values
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            
            return json_str
        except Exception as e:
            print(f"Error in JSON fixing: {e}")
            return json_str
    
    def _aggressive_json_fix(self, json_str: str) -> str:
        """More aggressive JSON fixing"""
        try:
            # Remove any explanatory text at the beginning
            lines = json_str.split('\n')
            json_lines = []
            found_start = False
            
            for line in lines:
                if not found_start and '{' in line:
                    found_start = True
                    # Take everything from the first { onwards
                    start_idx = line.find('{')
                    json_lines.append(line[start_idx:])
                elif found_start:
                    json_lines.append(line)
            
            if json_lines:
                json_str = '\n'.join(json_lines)
            
            # Final cleanup
            json_str = self._fix_common_json_issues(json_str)
            
            return json_str
        except Exception as e:
            print(f"Error in aggressive fixing: {e}")
            return json_str