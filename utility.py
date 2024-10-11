import json
import boto3
import random
import time
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
from retrying import retry

suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
aoss_client = boto3.client('opensearchserverless')
account_number = boto3.client('sts').get_caller_identity().get('Account')
identity = boto3.client('sts').get_caller_identity()['Arn']
credentials = boto3.Session().get_credentials()
bedrock_agent_client = boto3.client('bedrock-agent')


def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    oss_policy = iam_client.create_policy(
        PolicyName=oss_policy_name,
        PolicyDocument=json.dumps(oss_policy_document),
        Description='Policy for accessing opensearch serverless',
    )
    oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None


def create_access_policy_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return access_policy


def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)

def create_bedrock_execution_role_multi_ds(bucket_names):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1" 
                ]
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [item for sublist in [[f'arn:aws:s3:::{bucket}', f'arn:aws:s3:::{bucket}/*'] for bucket in bucket_names] for item in sublist], 
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role


def get_all_roles(iam_client):
    roles = []
    paginator = iam_client.get_paginator('list_roles')
    
    for page in paginator.paginate():
        roles.extend(page.get('Roles', []))
    
    return roles

def get_matching_roles(role_list, keyword):
    matching_roles = []
    for role in role_list:
        if isinstance(role, dict) and 'RoleName' in role and keyword in role['RoleName']:
            matching_roles.append(role)
    return matching_roles



def create_knowledge_base(index_name, collection_name, knowledge_base_name, vector_store_name, access_policy_name, embedding_model_arn):   
    
    body_json = {
       "settings": {
          "index.knn": "true",
           "number_of_shards": 1,
           "knn.algo_param.ef_search": 512,
           "number_of_replicas": 0,
       },
       "mappings": {
          "properties": {
             "vector": {
                "type": "knn_vector",
                "dimension": 1024,
                 "method": {
                     "name": "hnsw",
                     "engine": "faiss",
                     "space_type": "l2"
                 },
             },
             "text": {
                "type": "text"
             },
             "text-metadata": {
                "type": "text"         }
          }
       }
    }
    
    # Use the function
    all_roles = get_all_roles(iam_client)

    # Use the function
    matching_roles = get_matching_roles(all_roles, 'AmazonBedrockExecutionRoleForKnowledgeBases')

    bedrock_kb_execution_role_arn = matching_roles[0]["Arn"]

    # create data access policy within OSS
    try:
        response = aoss_client.list_access_policies(maxResults=100,type='data')
        # Check if the specific access policy is present in the response
        policy_exist = False
        access_policies = response.get('accessPolicySummaries', [])
        for policy in access_policies:
            if policy['name'] == access_policy_name:
                policy_exist = True
        if not policy_exist:
            print(f"Creating opensearch serverless access policy for vector store: {vector_store_name}\n")
            access_policy = create_access_policy_in_oss(vector_store_name=vector_store_name,
                        bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn,
                        access_policy_name=access_policy_name)
            interactive_sleep(60)
            print(f"Opensearch Serverless access policy creation for vector store: {vector_store_name} complete!!\n")
    except Exception as e: 
        print(f'Error while trying to create the access policy, with error {e}')

    
    # Build the OpenSearch client
    collections = aoss_client.list_collections()
    collection = [collection for collection in collections["collectionSummaries"] if collection["name"] == collection_name]
    awsauth = auth = AWSV4SignerAuth(credentials, region_name, 'aoss')
    host = collection[0]["id"] + '.' + region_name + '.aoss.amazonaws.com'
    oss_client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    
    # Create index
    try:
        indexExists = oss_client.indices.exists(index=index_name)
        if not indexExists:
            print(f"Creating index: {index_name}\n")
            response = oss_client.indices.create(index=index_name, body=json.dumps(body_json))

            # index creation can take up to a minute
            interactive_sleep(60)
            print(f"Index creation for index: {index_name} complete!!\n")
    except RequestError as e:
        # you can delete the index if its already exists
        # oss_client.indices.delete(index=index_name)
        print(f'Error while trying to create the index, with error {e.error}\nyou may unmark the delete above to delete, and recreate the index')
    
    
    opensearchServerlessConfiguration = {
            "collectionArn": collection[0]["arn"],
            "vectorIndexName": index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

    name = f"{knowledge_base_name}-knowledge-base"
    
    # Create a KnowledgeBase
    @retry(wait_random_min=1000, wait_random_max=2000,stop_max_attempt_number=7)
    def create_knowledge_base_func():
        create_kb_response = bedrock_agent_client.create_knowledge_base(
            name = name,
            roleArn = bedrock_kb_execution_role_arn,
            knowledgeBaseConfiguration = {
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": embedding_model_arn
                }
            },
            storageConfiguration = {
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration":opensearchServerlessConfiguration
            }
        )
        return create_kb_response["knowledgeBase"]
    try:
        response = bedrock_agent_client.list_knowledge_bases()
        knowledge_base_exist = False
        for kb in response['knowledgeBaseSummaries']:
            if kb['name'] == name:
                knowledge_base_found = True
                return kb
            
        if not knowledge_base_exist:
            print("Creating knowledge base...\n")
            kb = create_knowledge_base_func()
            print(f"Knowledge base: {kb['name']} created!!\n")
    except Exception as err:
        print(f"{err=}, {type(err)=}")
    
    return kb



def create_access_policy_in_oss(vector_store_name, bedrock_kb_execution_role_arn, access_policy_name):
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return access_policy


def create_ds(chunking_strategy_configuration, knowledge_base_id, bucket):
    
    s3_data_source_configuration = {
    "type": "S3",
    "s3Configuration": {
        "bucketArn": "",
        "inclusionPrefixes":["octank_financial_10K.pdf"] 
        }
    }

    data_sources=[
                {"type": "S3", "bucket_name": bucket} 
            ]
    
    
    ds_list=[]
    for idx, ds in enumerate(data_sources):

        if ds['type'] == "S3":
            ds_name = f'{knowledge_base_id}'
            s3_data_source_configuration["s3Configuration"]["bucketArn"] = f'arn:aws:s3:::{ds["bucket_name"]}'
            data_source_configuration = s3_data_source_configuration
        
        response = bedrock_agent_client.list_data_sources(knowledgeBaseId=knowledge_base_id)
        data_source_exist = False
        for data_source in response['dataSourceSummaries']:
            if data_source['name'] == knowledge_base_id:
                data_source_exist = True
                ds_list.append(data_source)
            
        if not data_source_exist:
            # Create a DataSource in KnowledgeBase 
            create_ds_response = bedrock_agent_client.create_data_source(
                name = ds_name,
                knowledgeBaseId = knowledge_base_id,
                dataSourceConfiguration = data_source_configuration,
                vectorIngestionConfiguration = {
                    "chunkingConfiguration": chunking_strategy_configuration
                }
            )
            ds = create_ds_response["dataSource"]
            ds_list.append(ds)
        
    ingest_jobs=[]
    # Start an ingestion job
    for idx, ds in enumerate(ds_list):
        try:
            start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = knowledge_base_id, dataSourceId = ds["dataSourceId"])
            job = start_job_response["ingestionJob"]
            print(f"Starting syncing job for Knowledge Base ID: {knowledge_base_id}\n")

            while job['status'] not in ["COMPLETE", "FAILED", "STOPPED"]:
                get_job_response = bedrock_agent_client.get_ingestion_job(
                  knowledgeBaseId = knowledge_base_id,
                    dataSourceId = ds["dataSourceId"],
                    ingestionJobId = job["ingestionJobId"]
              )
                job = get_job_response["ingestionJob"]
            print(f"Sync job for Knowledge Base ID: {knowledge_base_id} complete\n")

            ingest_jobs.append(job)
        except Exception as e:
            print(f"Couldn't start {idx} job.\n")
            print(e)  
    
    return ds_list