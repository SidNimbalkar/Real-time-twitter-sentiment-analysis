# Unit test cases for module functions present in ap.py 
import unittest
import ap

class Testap(unittest.TestCase):

    #checking of resulting dataframe has data
    def test_scraping(self):
        result = ap.scraping()
        self.assertIsNotNone(result)

    #checking if resulting dataframe has data 
    def test_scraping2(self):
        result = ap.scraping2()
        self.assertIsNotNone(result)
		
    def test_preprocess_tweet(self):
    #checking if preprocessing removes urls 
        text = "www.hello.com/hj hello we are doing great"
        result = ap.preprocess_tweet(text)
        self.assertEqual(result,"hello we are doing great")
    #checking if preprocessing removes urls
        text1 = "https://.newtest.com we are just testing here"
        result1 = ap.preprocess_tweet(text1)
        self.assertEqual(result1,"we are just testing here")
    #checking if preprocessing removes usernames 
        text2 = "@Rt we are testing @symbols"
        result2 = ap.preprocess_tweet(text2)
        self.assertEqual(result2,"we are testing")	
    #checking if preprocessing removes '#' symbols
        text3 = "my code #unit #testing"
        result3 = ap.preprocess_tweet(text3)
        self.assertEqual(result3,"my code unit testing")
		
if __name__ == '__main__':
    unittest.main()
