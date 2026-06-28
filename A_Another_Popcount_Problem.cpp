#include<bits/stdc++.h>
using namespace std;

int main(){
    int t;
    cin >> t;
    while(t--){
        int n, k;
        cin >> n >> k;

        int count = 0;
        int sum = 0;
        for(int p = 1; p <= n; p <<= 1){
            int take = min(k, n/p);

            n -= take*p;
            count += take;
        }

        cout << count << endl;
    }
}
