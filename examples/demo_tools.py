"""Synthetic calculator demonstration, not a task benchmark or model output."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shopping_agent.tools.voucher import calculate_public_bundle

def main():
    question='Voucher: same shop only, valid when total exceeds $15; 10% off capped at $10. Budget: $20.'
    products={'demo-a':{'product':{'price':7,'seller_id':'example-shop'}},
              'demo-b':{'product':{'price':7.28,'seller_id':'example-shop'}}}
    arguments={'product_ids':['demo-a','demo-b'],'voucher':{'scope':'same_shop','threshold':15,
               'threshold_operator':'>','discount_percent':10,'discount_cap':10},'budget':20}
    first=calculate_public_bundle(arguments,question,products)
    assert first['raw_subtotal']==14.28 and not first['threshold_met']
    products['demo-b']['product']['price']=8.01
    second=calculate_public_bundle(arguments,question,products)
    assert second['threshold_met'] and second['voucher_applicable']
    products['demo-b']['product']['seller_id']='another-example-shop'
    third=calculate_public_bundle(arguments,question,products)
    assert not third['shop_rule_met'] and not third['voucher_applicable']
    print(json.dumps({'synthetic_example':True,'below_threshold':first,'qualifies':second,'different_sellers':third},indent=2))
if __name__=='__main__':main()
