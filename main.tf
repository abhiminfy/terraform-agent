Terraform is an open-source infrastructure as code (IaC) tool developed by HashiCorp.  It allows you to define and provision data center infrastructure using a declarative configuration language called HashiCorp Configuration Language (HCL), or optionally JSON.  

With Terraform, you describe your desired state of infrastructure (e.g., servers, networks, load balancers, databases), and Terraform figures out how to create, update, or delete resources to achieve that state.  It supports a wide range of cloud providers and other platforms through providers.  Key features include:

* **Infrastructure as Code:** Manage infrastructure through configuration files, enabling version control, collaboration, and automation.
* **Declarative Configuration:** Define the desired end state, and Terraform determines the necessary steps to reach it.
* **Execution Plans:** Preview changes before applying them to your infrastructure.
* **State Management:** Tracks the current state of your infrastructure for efficient updates and change management.
* **Modularity:** Use modules to encapsulate and reuse infrastructure components.
* **Extensibility:** Leverage providers to interact with various cloud providers, platforms, and services.