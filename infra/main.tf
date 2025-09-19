# ⚠️  DEMO CREDENTIALS - DO NOT USE IN PRODUCTION!
# Replace all demo credentials with real values before deploying
# Use a secrets manager for production secrets

resource "aws\_s3\_bucket" "logbucket" {
bucket = "tf-log-bucket"
}

resource "aws\_s3\_bucket\_versioning" "versioning" {
bucket = aws\_s3\_bucket.logbucket.id
versioning\_configuration {
status = "Enabled"
}
}

resource "aws\_s3\_bucket\_server\_side\_encryption\_configuration" "sse\_kms" {
bucket = aws\_s3\_bucket.logbucket.id
rule {
apply\_server\_side\_encryption\_by\_default {
sse\_algorithm     = "aws\:kms"
kms\_master\_key\_id = "alias/aws/s3"
}
}
}

resource "aws\_s3\_bucket\_lifecycle\_configuration" "lifecycle" {
bucket = aws\_s3\_bucket.logbucket.id
rule {
id     = "expire-oldlogs"
status = "Enabled"
filter {
prefix = "oldlogs/"
}
expiration {
days = 30
}
}
}

resource "aws\_s3\_bucket\_public\_access\_block" "block\_public\_access" {
bucket                  = aws\_s3\_bucket.logbucket.id
block\_public\_acls       = true
block\_public\_policy     = true
ignore\_public\_acls      = true
restrict\_public\_buckets = true
}

resource "aws\_s3\_bucket\_policy" "deny\_insecure\_put" {
bucket = aws\_s3\_bucket.logbucket.id
policy = jsonencode({
Version = "2012-10-17"
Statement = \[{
Sid       = "DenyInsecurePut"
Effect    = "Deny"
Principal = "*"
Action    = "s3\:PutObject"
Resource  = "arn\:aws\:s3:::\${aws\_s3\_bucket.logbucket.bucket}/*"
Condition = {
Bool = { "aws\:SecureTransport" = "false" }
}
}]
})
}
