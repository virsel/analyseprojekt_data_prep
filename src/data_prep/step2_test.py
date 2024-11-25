from data_prep.step3 import encode2tkids

# Example usage and testing function
def test_masking():
    """
    Tests the masking functionality with various cases.
    """
    test_cases = [
        "Check out this link http://example.com!",
        "@username mentioned something about $AAPL $9b",
        "Multiple $amzn $GOOGL stocks and @multiple @users",
        "Complex case: @user1 talks about $aapl at http://test.com with @user2",
        "Just plain text with no special elements",
        "https://t.co/something @handle $STOCK"
    ]
    
    print("Testing tweet masking:")
    print("-" * 50)
    for test in test_cases:
        print(f"Original: {test}")
        print(f"Masked:   {encode2tkids(test)}")
        print("-" * 50)
        
test_masking()