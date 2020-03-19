import javafx.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.sql.*;

public class tclTest {
    public static void main(String [] args){
//        test1();
//        test2();
//        test3();
        test7();
    }

    public static void test7(){
        String [] a = {"1","2"};
        MySQLDemo.main(a);
    }
    public static void test6(){
        class A implements Callable {
            @Override
            public Object call() throws Exception {
                System.out.println("Thead"+Thread.currentThread().getName());
//                try {
//                    Thread.sleep(2);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
                return 1;
            }
        }
        ExecutorService exec = Executors.newFixedThreadPool(3);
        Collection<Future<?>> tasks = new LinkedList<Future<?>>();

        A a = new A();
        Future<?> future = exec.submit(a);
        tasks.add(future);
        future = exec.submit(a);
        tasks.add(future);
        future = exec.submit(a);
        tasks.add(future);

        // wait for tasks completion
        for (Future<?> currTask : tasks) {
            try {
                System.out.println("result:"+ currTask.get());
            } catch (Throwable thrown) {
//                Logger.error(thrown, "Error while waiting for thread completion");
                System.out.println("waiting");
            }
        }
    }

    public static void test5(){
        class A implements Runnable{

            @Override
            public void run() {
                System.out.println("Thead"+Thread.currentThread().getName());
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
        ExecutorService exec = Executors.newFixedThreadPool(3);
        Collection<Future<?>> tasks = new LinkedList<Future<?>>();

        A a = new A();
        Future<?> future = exec.submit(a);
        tasks.add(future);
        future = exec.submit(a);
        tasks.add(future);
        future = exec.submit(a);
        tasks.add(future);

        // wait for tasks completion
        for (Future<?> currTask : tasks) {
            try {
                currTask.get();
                System.out.println(currTask.get());
                System.out.println(currTask);
            } catch (Throwable thrown) {
//                Logger.error(thrown, "Error while waiting for thread completion");
                System.out.println("waiting");
            }
        }
        System.out.println("END");
    }

    public static void test4(){
      Scanner sc = new Scanner(System.in);
      String a;
      while((a=sc.nextLine())!=null){
          for(String b :a.split("[ ]"))
             System.out.println(b);
      }
      Map map = new HashMap<String, String>();
//      InputStream

    }
    public static void test1(){
        File file = new File("D:/datasets\\Univariate_arff_results\\" +
                "DTW/Predictions/ItalyPowerDemand/testFold0.csv");
        System.out.println(file.exists());
        System.out.println(51/100+","+(int)1.9);
        List a = new ArrayList<Integer>();
        a.add(1);
        System.out.println("12\t"+a.get(0));
        File dir = new File("D:\\datasets\\Univariate_arff");
        File next[] = dir.listFiles();
        int count = 0;
        for(int i =0;i<next.length;i++){
            if(next[i].isDirectory())
                count++;
            System.out.println(next[i].getName());
        }
        System.out.println("count:"+count);
        String path= "D:\\datasets\\Univariate_arff";
        String[] aa = path.split("\\\\");
        for(int i= 0;i<aa.length;i++){
            System.out.println(aa[i]);
        }
        System.out.println(tclTest.class.toString().split(" ")[1]);
        String str="192.168.0.1";
        String[] strarray=str.split("[.]");
        for (int i = 0; i < strarray.length; i++)
            System.out.println(strarray[i]);

        File f = new File("tcl_test.txt");
        System.out.println(35+","+f.exists());
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(f,true));
            bw.write("hello\n");
            bw.flush();
            bw.close();
            BufferedReader br = new BufferedReader(new FileReader(f));
            String s;
            while((s = br.readLine())!=null){
                System.out.println(s==s.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Set set = new HashSet();
        set.add("a");
        if(set.contains("a")){
            System.out.println(54+""+set.contains("a"));
        }
        Pair<String,String> b =new Pair<>("hello","world");
        Pair<String,Pair> c =new Pair<>("helo",b);
        List<Pair> d = new ArrayList<>();
        d.add(b);
        d.add(c);
        List<List> e = new ArrayList<>();
        e.add(d);
        e.add(d);
        Pair<String,List> g =new Pair("1",e);
        System.out.println(g);
        File file2 = new File("D:/test3/test23/test24/test.txt");
        if(!file2.getParentFile().exists()){
            file2.getParentFile().mkdirs();
        }
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file2,true));
            bw.write("test");
            bw.close();
        } catch (Exception e1) {
            e1.printStackTrace();
        }
        File file3 = new File("D:/test3/test23/");
        if(file3.exists()){
            file3.mkdirs();
            System.out.println("82："+file3.listFiles());
        }
        System.out.println(file3.getAbsolutePath());
        System.out.println(file3.getAbsoluteFile());
        System.out.println(file3.getParent()+"/");
        File file4 = new File(file3.getParent()+"\\3.txt");
        System.out.println(file4);

        double[] msmParams = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0.01,
                0.01375,
                0.0175,
                0.02125,
                0.025,
                0.02875,
                0.0325,
                0.03625,
                0.04,
                0.04375,
                0.0475,
                0.05125,
                0.055,
                0.05875,
                0.0625,
                0.06625,
                0.07,
                0.07375,
                0.0775,
                0.08125,
                0.085,
                0.08875,
                0.0925,
                0.09625,
                0.1,
                0.136,
                0.172,
                0.208,
                0.244,
                0.28,
                0.316,
                0.352,
                0.388,
                0.424,
                0.46,
                0.496,
                0.532,
                0.568,
                0.604,
                0.64,
                0.676,
                0.712,
                0.748,
                0.784,
                0.82,
                0.856,
                0.892,
                0.928,
                0.964,
                1,
                1.36,
                1.72,
                2.08,
                2.44,
                2.8,
                3.16,
                3.52,
                3.88,
                4.24,
                4.6,
                4.96,
                5.32,
                5.68,
                6.04,
                6.4,
                6.76,
                7.12,
                7.48,
                7.84,
                8.2,
                8.56,
                8.92,
                9.28,
                9.64,
                10,
                13.6,
                17.2,
                20.8,
                24.4,
                28,
                31.6,
                35.2,
                38.8,
                42.4,
                46,
                49.6,
                53.2,
                56.8,
                60.4,
                64,
                67.6,
                71.2,
                74.8,
                78.4,
                82,
                85.6,
                89.2,
                92.8,
                96.4,
                100// </editor-fold>
        };
        for(int i=0;i<5;i++){
            System.out.println(194+":"+msmParams[i*100/5]);
        }
        for(int i=0;i<100;i++){
            System.out.println(197+":"+i+","+(int)Math.ceil(i/100.0));
            System.out.println(198+":"+i+","+(int)Math.floor(i/100.0)+1);
        }
    }

    public static void test2(){
        Properties capitals = new Properties();
        Set states;
        String str;

        capitals.put("Illinois.1.2","1");
        capitals.put("Illinois.1.2","2");
        capitals.put("Missouri", "Jefferson City");
        capitals.put("Washington", "Olympia");
        capitals.put("California", "Sacramento");
        capitals.put("Indiana", "Indianapolis");

        // Show all states and capitals in hashtable.
        states = capitals.keySet(); // get set-view of keys
        Iterator itr = states.iterator();
        try {
            FileOutputStream of = new FileOutputStream("test.properties");
            capitals.store(of, "commands");
            of.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        while(itr.hasNext()) {
            str = (String) itr.next();
            System.out.println("The capital of " +
                    str + " is " + capitals.getProperty(str) + ".");
        }
        System.out.println();

        // look for state not in list -- specify default
        str = capitals.getProperty("Florida", "Not Found");
        System.out.println("The capital of Florida is "
                + str + ".");
    }

    public static void test3(){
        try {
            FileInputStream ifi = new FileInputStream("test.properties");
            Properties pro = new Properties();
            pro.load(ifi);
            System.out.println(pro.keySet());
            for(Object s: pro.keySet()){
                System.out.println(pro.getProperty((String) s)+","+pro.getProperty((String) s).getClass());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}



class MySQLDemo {

//    // MySQL 8.0 以下版本 - JDBC 驱动名及数据库 URL
//    static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
//    static final String DB_URL = "jdbc:mysql://192.168.56.101:3308/jdbc";

    // MySQL 8.0 以上版本 - JDBC 驱动名及数据库 URL
    static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://192.168.56.101:3308/jdbc?useSSL=false&serverTimezone=UTC";


    // 数据库的用户名与密码，需要根据自己的设置
    static final String USER = "root";
    static final String PASS = "cs321";

    public static void main(String[] args) {
        Connection conn = null;
        Statement stmt = null;
        try{
            // 注册 JDBC 驱动
            Class.forName(JDBC_DRIVER);

            // 打开链接
            System.out.println("连接数据库...");
            conn = DriverManager.getConnection(DB_URL,USER,PASS);

            // 执行查询
            System.out.println(" 实例化Statement对象...");
            stmt = conn.createStatement();
            String sql;
            sql = "SELECT id, name, url FROM websites";
            ResultSet rs = stmt.executeQuery(sql);

            // 展开结果集数据库
            while(rs.next()){
                // 通过字段检索
                int id  = rs.getInt("id");
                String name = rs.getString("name");
                String url = rs.getString("url");

                // 输出数据
                System.out.print("ID: " + id);
                System.out.print(", 站点名称: " + name);
                System.out.print(", 站点 URL: " + url);
                System.out.print("\n");
            }
            // 完成后关闭
            rs.close();
            stmt.close();
            conn.close();
        }catch(SQLException se){
            // 处理 JDBC 错误
            se.printStackTrace();
        }catch(Exception e){
            // 处理 Class.forName 错误
            e.printStackTrace();
        }finally{
            // 关闭资源
            try{
                if(stmt!=null) stmt.close();
            }catch(SQLException se2){
            }// 什么都不做
            try{
                if(conn!=null) conn.close();
            }catch(SQLException se){
                se.printStackTrace();
            }
        }
        System.out.println("Goodbye!");
    }
}
