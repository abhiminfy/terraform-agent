resource "aws_instance" "example" {
  ami = "ami-09ac0b140f63d3458" # Amazon Linux 2 AMI (us-east-1) as of 2023-10-27. Check for latest
  instance_type = "t2.micro"
  subnet_id = data.aws_subnet.default.id

  tags = {
    Name = "example-ec2-instance"
  }
}


data "aws_subnet" "default" {
  availability_zone = "us-east-1a"
  default_for_az = true

}



resource "aws_security_group" "example" {
 name = "allow_ssh"
 description = "Allow SSH inbound traffic"

 ingress {
   from_port = 22
   to_port = 22
   protocol = "tcp"
   cidr_blocks = ["0.0.0.0/0"]
 }


  egress {
        from_port = 0
        to_port = 0
        protocol = "all"
        cidr_blocks = ["0.0.0.0/0"]

    }

}


resource "aws_security_group_association" "example" {

    security_group_id = aws_security_group.example.id

    vpc_id = data.aws_vpc.default.id



}


data "aws_vpc" "default" {
  default = true
}