resource "aws_s3_bucket" "website_bucket" {
  bucket = "my-tf-test-bucket-website"
  acl    = "private"

  website {
    index_document = "index.html"
    error_document = "error.html"
  }

  tags = {
    Name = "S3 bucket for static website hosting"
  }
}

resource "aws_s3_bucket_policy" "website_bucket_policy" {
  bucket = aws_s3_bucket.website_bucket.id
  policy = data.aws_iam_policy_document.s3_policy.json
}

data "aws_iam_policy_document" "s3_policy" {
  statement {
    sid       = "PublicReadGetObject"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.website_bucket.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceIp"

      values = ["0.0.0.0/0"] # Replace with specific IP ranges if necessary for enhanced security
    }


  }
}


resource "aws_cloudfront_distribution" "s3_distribution" {
  origin {
    domain_name = aws_s3_bucket.website_bucket.bucket_regional_domain_name
    origin_id   = "S3Origin"
    s3_origin_config {
 origin_access_identity = aws_cloudfront_origin_access_identity.origin_access_identity.cloudfront_access_identity_path
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"


  logging_config {
    include_cookies = false
    bucket          = "my-tf-test-bucket-website-logs.s3.amazonaws.com" # Replace with your desired S3 bucket name for logs
    prefix          = "cloudfront-logs/"
  }



  price_class = "PriceClass_All"

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3Origin"
    viewer_protocol_policy = "redirect-to-https"
 min_ttl                = 0
    default_ttl                = 3600
    max_ttl                = 86400
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

 tags = {
    Name = "CloudFront distribution for static website"
  }
}

resource "aws_cloudfront_origin_access_identity" "origin_access_identity" {
  comment = "OAI for CloudFront to access S3"
}


# Example usage: upload index.html to the S3 bucket after creating it.
resource "aws_s3_bucket_object" "index" {
  bucket = aws_s3_bucket.website_bucket.id
  key    = "index.html"
  source = "index.html" # Replace with your actual index.html file
  content_type = "text/html"

  # Ensure the object exists after the bucket is created
  depends_on = [aws_s3_bucket.website_bucket]
}

resource "aws_s3_bucket_object" "error" {
  bucket = aws_s3_bucket.website_bucket.id
  key    = "error.html"
  source = "error.html" # Replace with your actual error.html file
  content_type = "text/html"

    # Ensure the object exists after the bucket is created
  depends_on = [aws_s3_bucket.website_bucket]
}

# Create log bucket
resource "aws_s3_bucket" "log_bucket" {
 bucket = replace("my-tf-test-bucket-website-logs", "-", "_") # Must replace hyphens with underscores
  acl    = "private"
  force_destroy = true
  lifecycle {
    prevent_destroy = false
  }


}

resource "aws_s3_bucket_acl" "log_bucket_acl" {
  bucket = aws_s3_bucket.log_bucket.id
  acl = "log-delivery-write"
}



output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.s3_distribution.domain_name
}