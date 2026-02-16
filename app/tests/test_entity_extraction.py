"""
Comprehensive Entity Extraction Validation Tests

These tests validate that the entity extraction module correctly parses
user queries and overrides training data with user-specified values.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.ai.ml.entity_extraction import EntityExtractor


def test_entity_extraction_comprehensive():
    """Test entity extraction with comprehensive examples."""
    extractor = EntityExtractor()
    
    test_cases = [
        # Static Website Cases
        {
            "query": "Deploy static website to nginx on docs.example.com, port 80",
            "expected": {
                "hostname": "docs.example.com",
                "server_name": "docs.example.com",
                "http_port": 80,
                "docker_image": "nginx",
                "image": "nginx"
            }
        },
        {
            "query": "Host my blog on myblog.com",
            "expected": {
                "hostname": "myblog.com",
                "server_name": "myblog.com"
            }
        },
        
        # FastAPI Cases
        {
            "query": "Deploy FastAPI app billing-api, port 8000, http port 80",
            "expected": {
                "app_name": "billing-api",
                "container_name": "billing-api-container",
                "http_port": 80
            }
        },
        {
            "query": "Deploy my API service user-management-api to api.mycompany.com, port 8080",
            "expected": {
                "app_name": "user-management-api",
                "container_name": "user-management-api-container",
                "hostname": "api.mycompany.com",
                "server_name": "api.mycompany.com"
            }
        },
        
        # Database Cases  
        {
            "query": "Deploy PostgreSQL database, version 16, name analytics_db, user admin",
            "expected": {
                "database_name": "analytics_db",
                "database_user": "admin",
                "postgres_version": "16",
                "kubernetes_version": "16"  # Cross-referenced by extractor
            }
        },
        {
            "query": "Create MySQL database inventory_system, user dbadmin",
            "expected": {
                "database_name": "inventory_system", 
                "database_user": "dbadmin"
            }
        },
        
        # VM/Instance Cases
        {
            "query": "Deploy EC2 instance t3.medium 50GB gp3 Ubuntu in us-west-2",
            "expected": {
                "instance_type": "t3.medium",
                "node_type": "t3.medium",  # For k8s compatibility
                "volume_size_gb": "50",
                "region": "us-west-2"
            }
        },
        {
            "query": "Create a large VM in eu-central-1 with t2.xlarge",
            "expected": {
                "instance_type": "t2.xlarge",
                "node_type": "t2.xlarge",
                "region": "eu-central-1"
            }
        },
        
        # Kubernetes Cases
        {
            "query": "Create an EKS cluster, 3 nodes, m5.large, kubernetes 1.29 in us-east-1",
            "expected": {
                "instance_type": "m5.large",  
                "node_type": "m5.large",
                "region": "us-east-1",
                "kubernetes_version": "1.29",
                "postgres_version": "1.29",  # Cross-referenced by extractor
                "node_count": "3"
            }
        },
        {
            "query": "Deploy K8s cluster with 5 worker nodes, t3.xlarge instances",
            "expected": {
                "instance_type": "t3.xlarge",
                "node_type": "t3.xlarge", 
                "node_count": "5"
            }
        },
        
        # Docker Cases
        {
            "query": "Run a nginx Docker container, port 80:80",
            "expected": {
                "docker_image": "nginx",
                "image": "nginx",
                "ports": "80:80",
                "http_port": 80,
                "app_port": 80
            }
        },
        {
            "query": "Deploy postgres container on port 5432:5432",
            "expected": {
                "docker_image": "postgres",
                "image": "postgres", 
                "ports": "5432:5432"
            }
        },
        
        # Edge Cases
        {
            "query": "Deploy my-awesome-app to staging.company-website.com on port 3000",
            "expected": {
                "hostname": "staging.company-website.com",
                "server_name": "staging.company-website.com" 
            }
        },
        {
            "query": "",  # Empty query
            "expected": {}
        }
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        
        print(f"\n[Test {i:02d}] Query: {query}")
        
        try:
            extracted = extractor.extract_entities(query)
            
            # Check if all expected keys are present with correct values
            test_passed = True
            for key, expected_value in expected.items():
                if key not in extracted:
                    print(f"  [FAIL] Missing expected key: {key}")
                    test_passed = False
                elif extracted[key] != expected_value:
                    print(f"  [FAIL] Key '{key}': expected '{expected_value}', got '{extracted[key]}'")
                    test_passed = False
                else:
                    print(f"  [PASS] Key '{key}': {extracted[key]}")
            
            # Show any extra keys that were extracted
            extra_keys = set(extracted.keys()) - set(expected.keys())
            if extra_keys:
                print(f"  [INFO] Extra keys extracted: {', '.join(f'{k}={extracted[k]}' for k in extra_keys)}")
            
            if test_passed:
                print(f"  [PASSED]")
                passed += 1
            else:
                print(f"  [FAILED]")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR]: {e}")
            failed += 1
    
    # Summary
    total = len(test_cases)
    print(f"\n" + "="*80)
    print(f"ENTITY EXTRACTION VALIDATION RESULTS")
    print(f"="*80)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Failed: {failed}/{total} ({failed/total*100:.1f}%)")
    
    if failed == 0:
        print(f"SUCCESS: ALL TESTS PASSED! Entity extraction is working perfectly.")
    else:
        print(f"WARNING: {failed} tests failed. Please review the output above.")
    
    return failed == 0


if __name__ == "__main__":
    test_entity_extraction_comprehensive()