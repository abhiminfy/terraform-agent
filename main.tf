resource "aws_instance" "example" {
 ami           = data.aws_ami.amazon_linux_2_latest.id
 instance_type = "t2.micro"

  tags = {
    Name = "example-ec2-instance"
  }
}

data "aws_ami" "amazon_linux_2_latest" {
 most_recent = true
 owners      = ["amazon"]

 filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}