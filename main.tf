# Configure the AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2" # Replace with your desired region
}


# Create an execution role for the Lambda function
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
}

# Attach the AWSLambdaBasicExecutionRole policy to the Lambda execution role
resource "aws_iam_role_policy_attachment" "lambda_policy_attachment" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}


# Create the Lambda function
resource "aws_lambda_function" "example_lambda" {
 filename      = "lambda_function_payload.zip" # Replace with your zip file
 function_name = "example_lambda_function"
 role          = aws_iam_role.lambda_exec_role.arn
 handler       = "index.handler" # Replace with your handler
 runtime = "nodejs16.x" # Replace with your runtime

 source_code_hash = filebase64sha256("lambda_function_payload.zip") # Ensure to update whenever code changes

 # Example environment variables
 environment {
    variables = {
      KEY1 = "VALUE1"
 }
 }
}

# Create the API Gateway REST API
resource "aws_api_gateway_rest_api" "example_api" {
  name        = "example_api"
  description = "Example API Gateway"
}



# Create the API Gateway resource
resource "aws_api_gateway_resource" "proxy_resource" {
 rest_api_id = aws_api_gateway_rest_api.example_api.id
 parent_id   = aws_api_gateway_rest_api.example_api.root_resource_id
 path_part   = "{proxy+}"
}

# Create the API Gateway method
resource "aws_api_gateway_method" "proxy_method" {
  rest_api_id   = aws_api_gateway_rest_api.example_api.id
  resource_id   = aws_api_gateway_resource.proxy_resource.id
  http_method   = "ANY"
  authorization = "NONE"
}



# Create the API Gateway integration
resource "aws_api_gateway_integration" "lambda_integration" {
 rest_api_id             = aws_api_gateway_rest_api.example_api.id
 resource_id             = aws_api_gateway_resource.proxy_resource.id
 http_method             = aws_api_gateway_method.proxy_method.http_method
 integration_http_method = "POST"
 type                    = "aws_proxy"
 integration_subtype = "Event"


 integration_uri = aws_lambda_function.example_lambda.invoke_arn
}


# Create the API Gateway deployment
resource "aws_api_gateway_deployment" "example_deployment" {
  rest_api_id = aws_api_gateway_rest_api.example_api.id
  triggers = {
    redeployment = sha1(jsonencode(aws_api_gateway_integration.lambda_integration.id))
 }

 lifecycle {
    create_before_destroy = true
  }
}

# Create the API Gateway stage
resource "aws_api_gateway_stage" "example_stage" {
 deployment_id = aws_api_gateway_deployment.example_deployment.id
 rest_api_id   = aws_api_gateway_rest_api.example_api.id
 stage_name    = "dev"
}




# Grant API Gateway permission to invoke the Lambda function
resource "aws_lambda_permission" "api_gateway_permission" {
 statement_id  = "AllowAPIGatewayInvoke"
 action        = "lambda:InvokeFunction"
 function_name = aws_lambda_function.example_lambda.function_name
 principal     = "apigateway.amazonaws.com"

 source_arn = "${aws_api_gateway_rest_api.example_api.execution_arn}/*/*"

}

output "api_gateway_invoke_url" {
 value = aws_api_gateway_deployment.example_deployment.invoke_url
}