#!/bin/bash
# Deploy ClinSense API to AWS ECR + ECS Fargate
# Prerequisites: AWS CLI configured, Docker, ECR repo created

set -e

AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPO=${ECR_REPO:-clinsense-api}
ECS_CLUSTER=${ECS_CLUSTER:-clinsense}
ECS_SERVICE=${ECS_SERVICE:-clinsense-api}

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

echo "Building and pushing to ${ECR_URI}..."

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Create ECR repo if not exists
aws ecr describe-repositories --repository-names ${ECR_REPO} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

# Build and push
docker build -t ${ECR_REPO}:latest .
docker tag ${ECR_REPO}:latest ${ECR_URI}
docker push ${ECR_URI}

echo "Image pushed. Deploy to ECS:"
echo "  aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --force-new-deployment"
echo ""
echo "Or run as new task:"
echo "  aws ecs run-task --cluster ${ECS_CLUSTER} --task-definition clinsense-api --launch-type FARGATE ..."
