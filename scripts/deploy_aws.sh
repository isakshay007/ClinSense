#!/bin/bash
# Deploy ClinSense API to AWS ECR + ECS Fargate
# Prerequisites: AWS CLI configured, Docker, ECS cluster + service created
# Required IAM: ecsTaskExecutionRole with ECR pull, CloudWatch logs

set -e

AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPO=${ECR_REPO:-clinsense-api}
ECS_CLUSTER=${ECS_CLUSTER:-clinsense}
ECS_SERVICE=${ECS_SERVICE:-clinsense-api}
LOG_GROUP=${LOG_GROUP:-/ecs/clinsense}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DEF_FILE="${PROJECT_ROOT}/aws/task-definition.json"
TASK_DEF_OUT="/tmp/clinsense-task-def.json"

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

echo "=== ClinSense AWS ECS Deploy ==="
echo "Account: ${AWS_ACCOUNT} | Region: ${AWS_REGION}"
echo "ECR: ${ECR_URI}"
echo ""

# 1. Create CloudWatch log group if not exists
aws logs create-log-group --log-group-name "${LOG_GROUP}" --region ${AWS_REGION} 2>/dev/null || true

# 2. Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 3. Create ECR repo if not exists
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${AWS_REGION} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

# 4. Build and push Docker image
echo "Building Docker image..."
cd "${PROJECT_ROOT}"
docker build -t ${ECR_REPO}:latest .
docker tag ${ECR_REPO}:latest ${ECR_URI}
echo "Pushing to ECR..."
docker push ${ECR_URI}

# 5. Generate task definition with actual ECR URI and region
echo "Registering task definition..."
sed -e "s/ACCOUNT_ID/${AWS_ACCOUNT}/g" -e "s/REGION/${AWS_REGION}/g" "${TASK_DEF_FILE}" > "${TASK_DEF_OUT}"
aws ecs register-task-definition --cli-input-json "file://${TASK_DEF_OUT}" --region ${AWS_REGION} --no-cli-pager > /dev/null

# 6. Update ECS service if it exists
if aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region ${AWS_REGION} 2>/dev/null | grep -q "ACTIVE"; then
  echo "Updating ECS service..."
  aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --force-new-deployment --region ${AWS_REGION} --no-cli-pager > /dev/null
  echo "Deployment triggered. Check ECS console for status."
else
  echo "ECS service ${ECS_SERVICE} not found. Create it first:"
  echo "  aws ecs create-service --cluster ${ECS_CLUSTER} --service-name ${ECS_SERVICE} \\"
  echo "    --task-definition clinsense-api --desired-count 1 --launch-type FARGATE \\"
  echo "    --network-configuration 'awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}'"
fi

echo ""
echo "Done. API will be available at your load balancer or task IP."
