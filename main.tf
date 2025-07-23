resource "aws_instance" "example" {
  ami = "ami-09ac0b140f63d3458" # Amazon Machine Image ID
  instance_type = "t2.micro"
}