resource "aws_s3_bucket" "website_bucket" {
  bucket = "my-tf-test-bucket-website"
  acl    = "private"

 website {
    index_document = "index.html"
    error_document = "error.html"

  }

  policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-tf-test-bucket-website/*"
        }
    ]
}
EOF
}

resource "aws_s3_bucket_object" "index" {
  bucket = aws_s3_bucket.website_bucket.id
  key    = "index.html"
  source = "index.html" # Make sure this file exists in the same directory
 content_type = "text/html"
}


resource "aws_s3_bucket_object" "error" {
  bucket = aws_s3_bucket.website_bucket.id
  key    = "error.html"
  source = "error.html" # Make sure this file exists in the same directory
  content_type = "text/html"
}


output "website_endpoint" {
  value = aws_s3_bucket.website_bucket.website_endpoint
}