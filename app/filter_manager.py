"""Filter configuration and management system with comprehensive security."""

import yaml
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
import sys
import traceback
from types import FunctionType
import ast
import re
import hashlib
import time
import threading
from datetime import datetime, timedelta

from . import models
from .pending_votes import create_pending_vote_for_science_case
from .anomaly_service import anomaly_service

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Raised when code fails security validation."""
    pass

class CodeValidator:
    """Validates Python code for security issues before execution."""
    
    # Dangerous imports/modules that should never be allowed
    FORBIDDEN_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
        'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
        'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3', 'psycopg2',
        'mysql', 'pymongo', 'redis', 'memcache',
        'threading', 'multiprocessing', 'asyncio', 'concurrent',
        'ctypes', 'cffi', 'gc', 'weakref', 'inspect',
        'importlib', '__import__', 'eval', 'exec', 'compile',
        'open', 'file', 'input', 'raw_input',
        'exit', 'quit', 'reload', 'help',
        'vars', 'locals', 'globals', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
    }
    
    # Dangerous function calls
    FORBIDDEN_CALLS = {
        'exec', 'eval', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'exit', 'quit',
        'vars', 'locals', 'globals', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
    }
    
    # Dangerous attributes that should not be accessed
    FORBIDDEN_ATTRIBUTES = {
        '__import__', '__builtins__', '__globals__', '__locals__',
        '__dict__', '__class__', '__bases__', '__subclasses__',
        'func_globals', 'func_code', 'gi_frame', 'cr_frame'
    }
    
    @classmethod
    def validate_code(cls, code: str, function_name: str) -> Dict[str, Any]:
        """
        Validate Python code for security issues.
        
        Returns:
            Dict with 'valid': bool and 'errors': List[str]
        """
        errors = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
        except SyntaxError as e:
            return {'valid': False, 'errors': [f'Syntax error: {e}']}
        
        # Check for dangerous patterns
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names = []
                if isinstance(node, ast.Import):
                    module_names = [alias.name for alias in node.names]
                else:  # ImportFrom
                    if node.module:
                        module_names = [node.module]
                
                for module_name in module_names:
                    if any(forbidden in module_name for forbidden in cls.FORBIDDEN_IMPORTS):
                        errors.append(f'Forbidden import: {module_name}')
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    errors.append(f'Forbidden function call: {node.func.id}')
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in cls.FORBIDDEN_ATTRIBUTES:
                    errors.append(f'Forbidden attribute access: {node.attr}')
            
            # Check for loops without breaks (potential infinite loops)
            elif isinstance(node, (ast.While, ast.For)):
                has_break = any(isinstance(child, ast.Break) for child in ast.walk(node))
                if not has_break and not isinstance(node, ast.For):
                    errors.append('While loops must have explicit break conditions')
        
        # Check function definition
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if len(function_defs) != 1:
            errors.append('Code must contain exactly one function definition')
        elif function_defs[0].name != function_name:
            errors.append(f'Function name must be "{function_name}", got "{function_defs[0].name}"')
        else:
            func_def = function_defs[0]
            if len(func_def.args.args) != 2 or func_def.args.args[0].arg != 'ztfid' or func_def.args.args[1].arg != 'confidence_threshold':
                errors.append('Function must accept exactly two parameters: "ztfid" and "confidence_threshold"')
        
        # Check for return statements
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
        if not has_return:
            errors.append('Function must have at least one return statement')
        
        return {'valid': len(errors) == 0, 'errors': errors}

class SecureExecutor:
    """Executes Python code in a secure, sandboxed environment."""
    
    def __init__(self, timeout_seconds: int = 10, memory_limit_mb: int = 50):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
    
    def execute_function(self, compiled_function: FunctionType, ztfid: str, confidence_threshold: float) -> Dict[str, Any]:
        """
        Execute a compiled function with security restrictions.
        
        Returns:
            Result dictionary or raises SecurityError
        """
        result = {'error': None, 'result': None, 'execution_time': 0}
        start_time = time.time()
        
        try:
            # Use threading to implement timeout
            result_container = [None]
            exception_container = [None]
            
            def target():
                try:
                    result_container[0] = compiled_function(ztfid, confidence_threshold)
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout_seconds)
            
            if thread.is_alive():
                raise SecurityError(f'Function execution timed out after {self.timeout_seconds} seconds')
            
            if exception_container[0]:
                raise exception_container[0]
            
            result['result'] = result_container[0]
            result['execution_time'] = time.time() - start_time
            
            # Validate result format
            if not isinstance(result['result'], dict):
                raise SecurityError('Function must return a dictionary')
            
            if 'label' not in result['result'] or 'score' not in result['result']:
                raise SecurityError('Function must return dictionary with "label" and "score" keys')
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            raise SecurityError(f'Execution failed: {e}')

class FilterApprovalWorkflow:
    """Manages approval workflow for custom filters."""
    
    @staticmethod
    def create_approval_request(db: Session, filter_config: Dict[str, Any], 
                              requesting_user_id: int) -> models.ClassifierApprovalRequest:
        """Create a new approval request for a custom filter."""
        
        # Generate code hash for tracking
        code_hash = hashlib.sha256(
            filter_config.get('python_code', '').encode()
        ).hexdigest()
        
        approval_request = models.ClassifierApprovalRequest(
            classifier_name=filter_config.get('name', 'Unknown'),  # Database field name, but it's storing filter name
            science_case=filter_config.get('science_case', 'unknown'),
            python_code=filter_config.get('python_code', ''),
            code_hash=code_hash,
            confidence_threshold=filter_config.get('confidence_threshold', 50.0),
            description=filter_config.get('description', ''),
            url=filter_config.get('url', ''),
            requesting_user_id=requesting_user_id,
            status='pending',
            security_validated=False,
            created_at=datetime.utcnow()
        )
        
        db.add(approval_request)
        db.commit()
        return approval_request
    
    @staticmethod
    def approve_filter(db: Session, approval_id: int, approving_admin_id: int) -> bool:
        """Approve a filter for execution."""
        approval = db.query(models.ClassifierApprovalRequest).filter(
            models.ClassifierApprovalRequest.id == approval_id
        ).first()
        
        if not approval:
            return False
        
        # Validate the code again before approval
        validation_result = CodeValidator.validate_code(
            approval.python_code, 
            approval.classifier_name.lower().replace(' ', '_')  # Database field name, but it's the filter name
        )
        
        if not validation_result['valid']:
            approval.status = 'rejected'
            approval.rejection_reason = f"Security validation failed: {', '.join(validation_result['errors'])}"
            approval.reviewed_by = approving_admin_id
            approval.reviewed_at = datetime.utcnow()
            db.commit()
            return False
        
        approval.status = 'approved'
        approval.security_validated = True
        approval.reviewed_by = approving_admin_id
        approval.reviewed_at = datetime.utcnow()
        db.commit()
        return True

class FilterManager:
    """Manages filter configurations and execution with security."""
    
    def __init__(self, config_path: str = "app/filter_config.yaml"):
        """Initialize the filter manager."""
        self.config_path = Path(config_path)
        self.config = None
        self.compiled_functions = {}  # Cache for compiled custom functions
        self.executor = SecureExecutor()
        self.load_config()
    
    def load_config(self) -> bool:
        """Load filter configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Filter config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Clear compiled functions cache when config is reloaded
            self.compiled_functions = {}
            
            logger.info(f"Loaded filter configuration from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading filter config: {e}")
            return False
    
    def get_enabled_filters(self, science_case: str) -> List[Dict[str, Any]]:
        """Get list of enabled filters for a science case."""
        if not self.config:
            return []
        
        filters = self.config.get('filters', {}).get(science_case, [])
        return [f for f in filters if f.get('enabled', False)]
    
    def get_filter_info(self, science_case: str, filter_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific filter."""
        if not self.config:
            return None
        
        filters = self.config.get('filters', {}).get(science_case, [])
        for filter_def in filters:
            if filter_def.get('name') == filter_name:
                return filter_def
        
        return None
    
    def validate_and_compile_function(self, filter_config: Dict[str, Any]) -> Optional[FunctionType]:
        """
        Validate and compile a custom Python function with comprehensive security checks.
        
        Returns:
            Compiled function or None if validation fails
        """
        filter_name = filter_config.get('name', 'unknown')
        python_code = filter_config.get('python_code', '')
        
        if not python_code:
            logger.warning(f"No python_code found for filter {filter_name}")
            return None
        
        # Generate function name from filter name
        expected_func_name = re.sub(r'[^a-z0-9]', '_', filter_name.lower())
        
        # Validate code security
        validation_result = CodeValidator.validate_code(python_code, expected_func_name)
        if not validation_result['valid']:
            logger.error(f"Security validation failed for {filter_name}: {validation_result['errors']}")
            raise SecurityError(f"Code validation failed: {', '.join(validation_result['errors'])}")
        
        # Check if already compiled and cached
        cache_key = f"{filter_name}_{hashlib.sha256(python_code.encode()).hexdigest()}"
        if cache_key in self.compiled_functions:
            return self.compiled_functions[cache_key]
        
        try:
            # Create extremely restricted execution environment
            safe_globals = {
                '__builtins__': {
                    # Only the most basic built-ins
                    'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                    'dict': dict, 'list': list, 'tuple': tuple, 'set': set,
                    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                    'range': range, 'enumerate': enumerate, 'zip': zip,
                    # Math operations
                    'pow': pow,
                },
                # Provide safe math operations
                'math': type('math', (), {
                    'sqrt': lambda x: x ** 0.5,
                    'log': lambda x: __import__('math').log(x),
                    'exp': lambda x: __import__('math').exp(x),
                    'sin': lambda x: __import__('math').sin(x),
                    'cos': lambda x: __import__('math').cos(x),
                    'pi': 3.141592653589793,
                    'e': 2.718281828459045
                })(),
            }
            
            # Execute the code to define the function
            local_namespace = {}
            exec(python_code, safe_globals, local_namespace)
            
            # Find the compiled function
            compiled_function = local_namespace.get(expected_func_name)
            if not compiled_function or not callable(compiled_function):
                raise SecurityError(f"Could not find function '{expected_func_name}' in code")
            
            # Cache the compiled function
            self.compiled_functions[cache_key] = compiled_function
            logger.info(f"Successfully compiled and validated function for {filter_name}")
            return compiled_function
            
        except Exception as e:
            logger.error(f"Error compiling function for {filter_name}: {e}")
            raise SecurityError(f"Compilation failed: {e}")
    
    def execute_custom_filter(self, db: Session, filter_config: Dict[str, Any], 
                            ztfids: List[str]) -> List[Dict[str, Any]]:
        """
        Execute a custom filter function with full security validation.
        
        Returns:
            List of filtering results
        """
        filter_name = filter_config.get('name', 'unknown')
        
        # **SECURITY CHECK: Only execute pre-approved filters**
        if not self._is_filter_approved(db, filter_config):
            logger.error(f"Filter {filter_name} is not approved for execution")
            raise SecurityError(f"Filter {filter_name} requires admin approval before execution")
        
        try:
            compiled_function = self.validate_and_compile_function(filter_config)
            if not compiled_function:
                return []
            
            results = []
            confidence_threshold = filter_config.get('confidence_threshold', 50.0)
            
            # Log execution attempt
            logger.info(f"Executing approved filter {filter_name} on {len(ztfids)} objects")
            
            for ztfid in ztfids:
                try:
                    # Execute with security restrictions
                    execution_result = self.executor.execute_function(compiled_function, ztfid, confidence_threshold)
                    result = execution_result['result']
                    
                    # Check if result meets confidence threshold
                    score = float(result.get('score', 0))
                    if score >= confidence_threshold:
                        results.append({
                            'ztfid': ztfid,
                            'filter': filter_name,
                            'label': result['label'],
                            'score': score,
                            'threshold': confidence_threshold,
                            'execution_time': execution_result['execution_time']
                        })
                        logger.info(f"{filter_name} filtered {ztfid} as {result['label']} with {score:.1f}% confidence")
                
                except SecurityError as e:
                    logger.error(f"Security error executing {filter_name} for {ztfid}: {e}")
                    # Log security violations for audit
                    self._log_security_violation(db, filter_name, ztfid, str(e))
                    continue
                except Exception as e:
                    logger.error(f"Error executing {filter_name} for {ztfid}: {e}")
                    continue
            
            return results
            
        except SecurityError as e:
            logger.error(f"Security validation failed for {filter_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing {filter_name}: {e}")
            raise SecurityError(f"Execution failed: {e}")
    
    def _is_filter_approved(self, db: Session, filter_config: Dict[str, Any]) -> bool:
        """Check if a filter has been approved by an admin."""
        # For now, only allow the built-in reLAISS filter
        # Custom filters would need to go through approval workflow
        if filter_config.get('name') == 'reLAISS':
            return True
        
        # For custom filters, check approval status
        code_hash = hashlib.sha256(
            filter_config.get('python_code', '').encode()
        ).hexdigest()
        
        approval = db.query(models.ClassifierApprovalRequest).filter(
            models.ClassifierApprovalRequest.code_hash == code_hash,
            models.ClassifierApprovalRequest.status == 'approved',
            models.ClassifierApprovalRequest.security_validated == True
        ).first()
        
        return approval is not None
    
    def _log_security_violation(self, db: Session, filter_name: str, ztfid: str, error: str):
        """Log security violations for audit purposes."""
        try:
            violation = models.SecurityAuditLog(
                event_type='filter_security_violation',
                classifier_name=filter_name,  # Database field name, but storing filter name
                ztfid=ztfid,
                error_message=error,
                timestamp=datetime.utcnow(),
                severity='HIGH'
            )
            db.add(violation)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log security violation: {e}")
    
    def run_filters_for_science_case(self, db: Session, science_case: str, 
                                   feature_extraction_run: models.FeatureExtractionRun) -> List[str]:
        """
        Run all enabled filters for a science case with security checks.
        
        Returns:
            List of ZTFIDs that were filtered as pending for this science case
        """
        if science_case == "anomalous":
            # Special handling for anomalous - use existing anomaly service
            notification = anomaly_service.process_new_objects_for_anomalies(db, feature_extraction_run)
            if notification and notification.ztfids_detected:
                return notification.ztfids_detected
            return []
        
        enabled_filters = self.get_enabled_filters(science_case)
        if not enabled_filters:
            logger.info(f"No enabled filters for {science_case}")
            return []
        
        # Get all ZTFIDs from the feature extraction run
        feature_objects = db.query(models.FeatureBank).filter(
            models.FeatureBank.mjd_extracted >= feature_extraction_run.mjd_run - 0.1
        ).all()
        
        if not feature_objects:
            logger.info(f"No feature objects found for recent extraction run")
            return []
        
        ztfids = [obj.ztfid for obj in feature_objects]
        logger.info(f"Running {len(enabled_filters)} filters on {len(ztfids)} objects for {science_case}")
        
        filtered_objects = []
        
        for filter_config in enabled_filters:
            try:
                filter_name = filter_config.get('name', 'unknown')
                
                # **SECURITY: Only execute approved filters**
                if not self._is_filter_approved(db, filter_config):
                    logger.warning(f"Skipping unapproved filter {filter_name}")
                    continue
                
                if filter_config.get('python_code'):
                    # Custom Python code filter with security validation
                    results = self.execute_custom_filter(db, filter_config, ztfids)
                    
                    # Create pending votes for filtered objects
                    for result in results:
                        try:
                            pending_vote = create_pending_vote_for_science_case(
                                db=db,
                                ztfid=result['ztfid'],
                                science_case=science_case,
                                details={
                                    'filter_name': filter_name,
                                    'confidence': result['score'],
                                    'label': result['label'],
                                    'threshold': result['threshold'],
                                    'execution_time': result.get('execution_time', 0)
                                }
                            )
                            if pending_vote:
                                filtered_objects.append(result['ztfid'])
                                logger.info(f"Created pending vote for {result['ztfid']} in {science_case}")
                        
                        except Exception as e:
                            logger.error(f"Error creating pending vote for {result['ztfid']}: {e}")
                
                else:
                    # Legacy module/function based filter (trusted)
                    module_name = filter_config.get('module')
                    function_name = filter_config.get('function')
                    
                    if module_name and function_name:
                        try:
                            module = importlib.import_module(module_name)
                            filter_function = getattr(module, function_name)
                            results = filter_function(db, feature_extraction_run)
                            
                            if results:
                                filtered_objects.extend(results)
                                logger.info(f"Legacy filter {filter_name} filtered {len(results)} objects")
                        
                        except (ImportError, AttributeError) as e:
                            logger.warning(f"Could not load legacy filter {filter_name}: {e}")
                
            except SecurityError as e:
                logger.error(f"Security error with filter {filter_config.get('name', 'unknown')}: {e}")
                # Continue with other filters even if one fails security checks
                continue
            except Exception as e:
                logger.error(f"Error running filter {filter_config.get('name', 'unknown')}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Completed {science_case} filtering: {len(filtered_objects)} objects filtered")
        return filtered_objects
    
    def create_placeholder_pending_votes(self, db: Session) -> None:
        """
        Create placeholder pending votes for all science cases except anomalous.
        This ensures the system works out of the box with empty pending lists.
        """
        science_cases = ["snia-like", "ccsn-like", "long-lived", "precursor"]
        
        for science_case in science_cases:
            try:
                # Check if there are already pending votes for this science case
                existing_pending = db.query(models.PendingVote).filter(
                    models.PendingVote.science_case == science_case
                ).first()
                
                if not existing_pending:
                    logger.info(f"No pending votes found for {science_case} - system ready for future filters")
                    # Note: We don't create dummy pending votes, just ensure the system can handle empty lists
                
            except Exception as e:
                logger.error(f"Error checking pending votes for {science_case}: {e}")
    
    def get_filter_badge_info(self, db: Session, ztfid: str) -> List[Dict[str, Any]]:
        """
        Get filter badge information for a ZTFID.
        
        Returns:
            List of badge information with filter name, confidence, description, and URL
        """
        badges = []
        
        # Check for anomaly detection results
        anomaly_result = db.query(models.AnomalyDetectionResult).filter(
            models.AnomalyDetectionResult.ztfid == ztfid
        ).first()
        
        if anomaly_result and anomaly_result.is_anomalous:
            filter_info = self.get_filter_info("anomalous", "reLAISS")
            if filter_info:
                # Count epochs with data
                num_epochs = len(anomaly_result.anom_scores) if anomaly_result.anom_scores else 0
                
                badges.append({
                    'filter_name': 'reLAISS',
                    'filter_url': filter_info['url'],
                    'confidence': anomaly_result.anomaly_score,
                    'num_epochs': num_epochs,
                    'description': filter_info['description'],
                    'badge_text': f"This transient was flagged by reLAISS with {anomaly_result.anomaly_score:.1f}% anomaly score across {num_epochs} epochs.",
                    'badge_type': 'anomaly'
                })
        
        # Check for custom filter results
        custom_results = db.query(models.CustomClassifierResult).filter(
            models.CustomClassifierResult.ztfid == ztfid
        ).all()
        
        for result in custom_results:
            filter_info = self.get_filter_info(result.science_case, result.classifier_name)  # Database field name, but it's the filter name
            if filter_info:
                badges.append({
                    'filter_name': result.classifier_name,
                    'filter_url': filter_info.get('url', '#'),
                    'confidence': result.confidence_score,
                    'classification': result.classification_label,
                    'description': filter_info.get('description', 'Custom filter'),
                    'badge_text': f"This transient was filtered by {result.classifier_name} as {result.classification_label} with {result.confidence_score:.1f}% confidence.",
                    'badge_type': 'classification'
                })
        
        return badges

# Global instance
filter_manager = FilterManager() 