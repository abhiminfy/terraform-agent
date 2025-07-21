# Create an execution role for the Lambda function
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_exec_role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

# Attach the AWSLambdaBasicExecutionRole policy to the role
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Create a policy to allow logging to CloudWatch Logs
resource "aws_iam_policy" "lambda_cloudwatch_policy" {
  name = "lambda_cloudwatch_policy"

 policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
EOF
}


# Attach the CloudWatch logging policy to the role
resource "aws_iam_role_policy_attachment" "lambda_cloudwatch_attachment" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_cloudwatch_policy.arn
}



# Create the Lambda function
resource "aws_lambda_function" "example" {
  filename      = null # No filename since we're using inline code
  function_name = "my_lambda_function"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "index.handler"
  runtime = "nodejs18.x"

  source_code_hash = base64sha256(data.archive_file.lambda_zip.output_base64sha256)
  memory_size = 128
  timeout = 30


  data.archive_file.lambda_zip.output_path


  data "archive_file" "lambda_zip" {
    type        = "zip"
    output_path = "${path.module}/lambda_function.zip"
    source {
      content  = <<EOF
exports.handler = async (event) => {
  console.log("Event:", JSON.stringify(event, null, 2));
  return {
    statusCode: 200,
    body: JSON.stringify({ message: "Hello from Lambda!" }),
  };
};
EOF
      filename = "index.js"
    }
  }


}