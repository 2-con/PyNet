"""
GenAI
=====

"""

class Client:
  def __init__(self, model, apikey):
    self.model = model
    self.valid_request = apikey in database.VALID_API_KEYS

    if not self.valid_request:
      raise ValueError("Invalid API key - request blocked")

  def generate(self, content):
    # generate content using the model
    pass

  def summary(self, show_sensitive_content=False):
    print( "Model Summary:")
    print("-------------------------------------------------------------")
    print( "API        : AufyAI (default)")
    print(f"Model      : {self.model}")
    print(f"Key Status : {'Valid' if self.valid_request else 'Invalid'}")
    print(f"API Key    : {self.apikey if show_sensitive_content else '[Hidden]'}")
    print("-------------------------------------------------------------")