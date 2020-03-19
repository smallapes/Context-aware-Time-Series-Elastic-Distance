class b{
    public b(){
        System.out.println("B---");
    }
}
public class a implements Runnable{
    public a(){
        System.out.println("end");
    }
    public void init(){
        System.out.println("---init----");
    }
    public static void  main(String [] args){
        new Thread(new a()).start();
        System.out.println();
    }

    @Override
    public void run() {

    }

}