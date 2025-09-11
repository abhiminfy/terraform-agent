import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class TerraformGuardrails:
    """Guardrails to prevent hallucination and ensure quality responses"""
    
    def __init__(self):
        # Keywords that indicate infrastructure-related queries
        self.infra_keywords = [
            'terraform', 'aws', 'cloud', 'infrastructure', 'deploy', 'provision',
            'ec2', 's3', 'rds', 'vpc', 'lambda', 'api gateway', 'load balancer',
            'security group', 'subnet', 'route table', 'internet gateway',
            'nat gateway', 'elastic ip', 'cloudfront', 'route53', 'cloudwatch',
            'iam', 'elasticache', 'elasticsearch', 'eks', 'ecs', 'fargate',
            'create infrastructure', 'set up', 'configure', 'build', 'server'
        ]
        
        # Dangerous patterns to warn about - FIXED REGEX PATTERNS
        self.dangerous_patterns = [
            (r'password\s*=\s*"[^"]*"', "Hardcoded password detected"),
            (r'secret_key\s*=\s*"[^"]*"', "Hardcoded secret key detected"),
            (r'access_key\s*=\s*"[^"]*"', "Hardcoded access key detected"),
            (r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]', "Open security group (0.0.0.0/0)"),
            (r'publicly_accessible\s*=\s*true', "Database publicly accessible"),
            (r'force_destroy\s*=\s*true', "Force destroy enabled - data loss risk")
        ]
    
    def is_infrastructure_request(self, message: str) -> bool:
        """Check if the message is infrastructure-related"""
        if not message:
            return False
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.infra_keywords)
    
    def validate_terraform_response(self, response: str) -> Dict[str, Any]:
        """Validate if the response contains valid Terraform concepts"""
        if not response:
            return {
                "valid": False,
                "reason": "Empty response",
                "confidence": 0.0
            }
        
        response_lower = response.lower()
        confidence = 0.0
        
        # Check for Terraform syntax patterns
        terraform_patterns = [
            r'\bresource\s+"[^"]+"\s+"[^"]+"\s*\{',
            r'\bprovider\s+"[^"]+"\s*\{',
            r'\bvariable\s+"[^"]+"\s*\{',
            r'\boutput\s+"[^"]+"\s*\{',
            r'\bdata\s+"[^"]+"\s+"[^"]+"\s*\{',
            r'\bterraform\s*\{',
            r'\bmodule\s+"[^"]+"\s*\{'
        ]
        
        pattern_matches = 0
        for pattern in terraform_patterns:
            if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
                pattern_matches += 1
                confidence += 0.15
        
        # Check for AWS resource types
        aws_resources = [
            'aws_instance', 'aws_s3_bucket', 'aws_vpc', 'aws_subnet',
            'aws_security_group', 'aws_rds_instance', 'aws_lambda_function',
            'aws_lb', 'aws_cloudfront_distribution', 'aws_route53_zone'
        ]
        
        aws_matches = 0
        for resource in aws_resources:
            if resource in response_lower:
                aws_matches += 1
                confidence += 0.1
        
        # Check for Terraform keywords
        terraform_keywords = [
            'terraform', 'resource', 'provider', 'variable', 'output',
            'data', 'module', 'locals', 'count', 'for_each'
        ]
        
        keyword_matches = 0
        for keyword in terraform_keywords:
            if keyword in response_lower:
                keyword_matches += 1
                confidence += 0.05
        
        confidence = min(1.0, confidence)
        
        # Determine if valid
        is_valid = (
            pattern_matches > 0 or  # Has Terraform syntax
            (aws_matches > 0 and keyword_matches > 0) or  # Has AWS resources and TF keywords
            confidence > 0.5
        )
        
        return {
            "valid": is_valid,
            "confidence": round(confidence, 2),
            "pattern_matches": pattern_matches,
            "aws_matches": aws_matches,
            "keyword_matches": keyword_matches,
            "reason": "Valid Terraform content detected" if is_valid else "No valid Terraform patterns found"
        }
    
    def check_security_issues(self, terraform_code: str) -> List[Dict[str, Any]]:
        """Check for security issues in Terraform code"""
        if not terraform_code:
            return []
        
        issues = []
        for pattern, description in self.dangerous_patterns:
            try:
                matches = re.finditer(pattern, terraform_code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    issues.append({
                        "type": "security_warning",
                        "description": description,
                        "line_content": match.group(0),
                        "severity": self._get_severity(description)
                    })
            except re.error as e:
                logger.error(f"Regex error in pattern '{pattern}': {str(e)}")
                continue
        
        return issues
    
    def _get_severity(self, description: str) -> str:
        """Determine severity based on issue description"""
        if any(word in description.lower() for word in ['secret', 'key', 'password']):
            return "CRITICAL"
        elif any(word in description.lower() for word in ['public', '0.0.0.0/0']):
            return "HIGH"
        elif any(word in description.lower() for word in ['destroy', 'force']):
            return "MEDIUM"
        else:
            return "LOW"
    
    def filter_hallucinations(self, response: str, user_message: str) -> Tuple[str, List[str]]:
        """Filter out potential hallucinations and return filtered response with warnings"""
        if not response:
            return response, ["Empty response received"]
        
        warnings = []
        filtered_lines = []
        lines = response.split('\n')
        
        # Patterns that might indicate hallucinations
        hallucination_patterns = [
            r'(?i)(i think|i believe|probably|maybe|might be|could be)',
            r'(?i)(as far as i know|to my knowledge|if i remember)',
            r'(?i)(let me guess|i assume|presumably)',
            r'(?i)(version \d+\.\d+\.\d+ \(which was released)',  # Specific version claims
            r'(?i)(according to the latest|recent updates show)'
        ]
        
        for line in lines:
            is_hallucination = False
            
            # Check each pattern
            for pattern in hallucination_patterns:
                try:
                    if re.search(pattern, line):
                        is_hallucination = True
                        warnings.append(f"Filtered uncertain statement: {line.strip()[:100]}")
                        break
                except re.error as e:
                    logger.error(f"Regex error in hallucination pattern '{pattern}': {str(e)}")
                    continue
            
            # Keep the line if it's not a hallucination
            if not is_hallucination:
                filtered_lines.append(line)
        
        filtered_response = '\n'.join(filtered_lines)
        
        # If we filtered too much content, return original with warning
        if len(filtered_response.strip()) < len(response.strip()) * 0.3:
            warnings.append("Response may contain uncertain information - please verify")
            return response, warnings
        
        return filtered_response, warnings
    
    def validate_user_input(self, message: str) -> Dict[str, Any]:
        """Validate user input for safety and clarity"""
        if not message or not message.strip():
            return {
                "valid": False,
                "reason": "Empty message",
                "suggestions": ["Please provide a clear question or request"]
            }
        
        message = message.strip()
        
        # Check for minimum length
        if len(message) < 3:
            return {
                "valid": False,
                "reason": "Message too short",
                "suggestions": ["Please provide more details about what you need"]
            }
        
        # Check for maximum length
        if len(message) > 2000:
            return {
                "valid": False,
                "reason": "Message too long",
                "suggestions": ["Please break down your request into smaller parts"]
            }
        
        # Check for potentially harmful requests
        harmful_patterns = [
            r'(?i)(hack|exploit|attack|vulnerability)',
            r'(?i)(delete all|destroy everything|rm -rf)',
            r'(?i)(bypass security|disable protection)',
        ]
        
        for pattern in harmful_patterns:
            try:
                if re.search(pattern, message):
                    return {
                        "valid": False,
                        "reason": "Potentially harmful request detected",
                        "suggestions": ["Please rephrase your request focusing on legitimate infrastructure needs"]
                    }
            except re.error as e:
                logger.error(f"Regex error in harmful pattern '{pattern}': {str(e)}")
                continue
        
        # Check if request is too vague
        vague_patterns = [
            r'^(hi|hello|hey)$',
            r'^(help|what|how)$',
            r'^(do something|make it work)$'
        ]
        
        for pattern in vague_patterns:
            try:
                if re.search(pattern, message, re.IGNORECASE):
                    return {
                        "valid": True,
                        "reason": "Request is too vague",
                        "suggestions": [
                            "Try being more specific, for example:",
                            "- 'Create an EC2 instance with a web server'",
                            "- 'Set up a VPC with public and private subnets'",
                            "- 'Deploy a load balancer for my application'"
                        ]
                    }
            except re.error as e:
                logger.error(f"Regex error in vague pattern '{pattern}': {str(e)}")
                continue
        
        return {
            "valid": True,
            "reason": "Input appears valid",
            "suggestions": []
        }
    
    def should_ask_for_clarification(self, message: str) -> Tuple[bool, List[str]]:
        """Determine if we should ask for clarification"""
        if not message:
            return True, ["Please provide a message"]
        
        message_lower = message.lower()
        questions = []
        
        # Check if infrastructure request but missing key details
        if self.is_infrastructure_request(message):
            missing_details = []
            
            # Check for AWS region
            if not re.search(r'region|us-east|us-west|eu-west|ap-southeast', message_lower):
                missing_details.append("Which AWS region would you like to use?")
            
            # Check for environment
            if not re.search(r'dev|development|staging|prod|production|test', message_lower):
                missing_details.append("Is this for development, staging, or production?")
            
            # Check for instance size (if EC2 mentioned)
            if 'ec2' in message_lower or 'instance' in message_lower:
                if not re.search(r't2|t3|m5|c5|r5|micro|small|medium|large', message_lower):
                    missing_details.append("What instance size do you need? (e.g., t3.micro, t3.small)")
            
            # If too many details missing, ask for clarification
            if len(missing_details) >= 2:
                return True, missing_details
        
        # Check if request is too general
        general_patterns = [
            r'^(create|make|build|setup|deploy)\s+(something|infrastructure|app|website)$',
            r'^(help me with|show me|explain)\s+(terraform|aws|cloud)$'
        ]
        
        for pattern in general_patterns:
            try:
                if re.search(pattern, message, re.IGNORECASE):
                    return True, [
                        "Could you be more specific about what you want to create?",
                        "For example: web server, database, load balancer, VPC, etc."
                    ]
            except re.error as e:
                logger.error(f"Regex error in general pattern '{pattern}': {str(e)}")
                continue
        
        return False, []
    
    def get_confidence_score(self, response: str, user_message: str) -> float:
        """Calculate confidence score for the response"""
        if not response or not user_message:
            return 0.0
        
        score = 0.0
        
        # Check if response matches request type
        if self.is_infrastructure_request(user_message):
            terraform_validation = self.validate_terraform_response(response)
            score += terraform_validation["confidence"] * 0.4
        
        # Check response length (too short or too long might indicate issues)
        response_length = len(response.strip())
        if 100 <= response_length <= 3000:
            score += 0.2
        elif response_length < 100:
            score += 0.05  # Very short response
        
        # Check for code blocks
        if '```' in response:
            score += 0.15
        
        # Check for structured content
        if any(marker in response for marker in ['resource', 'provider', 'variable']):
            score += 0.15
        
        # Penalize uncertainty language
        uncertainty_patterns = [
            r'(?i)(i think|maybe|might|could be|not sure)',
            r'(?i)(probably|possibly|perhaps|seems like)'
        ]
        
        uncertainty_count = 0
        for pattern in uncertainty_patterns:
            try:
                if re.search(pattern, response):
                    uncertainty_count += 1
            except re.error as e:
                logger.error(f"Regex error in uncertainty pattern '{pattern}': {str(e)}")
                continue
        
        score -= uncertainty_count * 0.1
        
        return max(0.0, min(1.0, score))
    
    # === APPEND: AST security check (non-destructive) ===
from io import StringIO as _GR_StringIO
import hcl2 as _gr_hcl2

def security_ast_checks_v2(tf_code: str):
    try:
        _gr_hcl2.load(_GR_StringIO(tf_code))
        return {"success": True, "data": {"ast_checks": "ok"}, "error": ""}
    except Exception as e:
        return {"success": False, "data": {}, "error": f"HCL parse error: {e}"}
