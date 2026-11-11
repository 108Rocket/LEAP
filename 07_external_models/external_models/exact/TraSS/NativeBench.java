import com.just.ksim.disks.Client;
import com.just.ksim.entity.Trajectory;
import com.just.ksim.filter.CalculateSimilarity;
import com.just.ksim.similarity.Frechet;
import com.just.ksim.similarity.Hausdorff;
import com.just.ksim.similarity.DTW;
import com.google.common.collect.MinMaxPriorityQueue;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.*;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.filter.FirstKeyOnlyFilter;
import org.apache.hadoop.hbase.util.Bytes;
import org.locationtech.jts.geom.*;
import scala.Tuple2;
import utils.PutUtils;
import utils.WKTUtils;
import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Benchmark adapter; native index, DP features, filters and scorers are retained. */
public class NativeBench {
    static final int WORKERS=12, K=20;
    static final short G=16, SHARDS=8;
    static final double INTERVAL=.002;
    static final String CF=utils.Constants.DEFAULT_CF;
    static final GeometryFactory GF=new GeometryFactory(new PrecisionModel(),4326);

    static class Npy implements Closeable {
        final FileChannel ch;
        final ByteBuffer buf;
        final int offset;
        Npy(Path p, String dtype) throws Exception {
            ch=FileChannel.open(p, StandardOpenOption.READ);
            if(ch.size()>Integer.MAX_VALUE) throw new IOException("NPY mapping exceeds 2GB: "+p);
            buf=ch.map(FileChannel.MapMode.READ_ONLY,0,ch.size()).order(ByteOrder.LITTLE_ENDIAN);
            if(buf.get(0)!=(byte)0x93) throw new IOException("Not NPY");
            int version=buf.get(6);
            int len=version==1?Short.toUnsignedInt(buf.getShort(8)):buf.getInt(8);
            offset=(version==1?10:12)+len;
            byte[] h=new byte[len];
            ByteBuffer copy=buf.duplicate();copy.position(version==1?10:12);copy.get(h);
            String header=new String(h,"UTF-8");
            if(!header.contains(dtype)||!header.contains("False")) throw new IOException(header);
        }
        long lng(long i){return buf.getLong(Math.toIntExact(offset+8*i));}
        double dbl(long i){return buf.getDouble(Math.toIntExact(offset+8*i));}
        public void close() throws IOException {ch.close();}
    }
    static class Dataset implements Closeable {
        final Npy pts,offs; final int n;
        Dataset(Path root,String prefix)throws Exception {
            pts=new Npy(root.resolve(prefix+"_points.npy"),"<f8");
            offs=new Npy(root.resolve(prefix+"_offsets.npy"),"<i8");
            n=(int)((offs.ch.size()-offs.offset)/8)-1;
        }
        Trajectory get(int id){
            long start=offs.lng(id),end=offs.lng(id+1);
            Coordinate[] c=new Coordinate[(int)(end-start)];
            for(int i=0;i<c.length;i++) c[i]=new Coordinate(pts.dbl(2*(start+i)),pts.dbl(2*(start+i)+1));
            return new Trajectory(Integer.toString(id),GF.createMultiPointFromCoords(c));
        }
        public void close()throws IOException{pts.close();offs.close();}
    }
    static void save(Path p,Object obj)throws Exception {
        // Jackson is supplied by the author's/HBase runtime, not a hand-rolled JSON encoder.
        new com.fasterxml.jackson.databind.ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(p.toFile(),obj);
    }
    static Map<String,Object> row(Object... kv){
        Map<String,Object> m=new LinkedHashMap<>();
        for(int i=0;i<kv.length;i+=2)m.put(kv[i].toString(),kv[i+1]);return m;
    }
    static int[] ids(Path p,int n)throws Exception{
        if(p==null){int[] x=new int[n];for(int i=0;i<n;i++)x[i]=i;return x;}
        return Files.lines(p).filter(s->!s.trim().isEmpty()).mapToInt(Integer::parseInt).toArray();
    }
    static void feature(Trajectory t){t.getDPFeature().getIndexes();t.getDPFeature().getMBRs();}
    static double distance(Trajectory a,Trajectory b,int f){
        if(f==0)return Frechet.calulateDistance(a.getMultiPoint(),b.getMultiPoint());
        if(f==1)return Hausdorff.calulateDistance(a.getMultiPoint(),b.getMultiPoint());
        return DTW.calulateDistance(a.getMultiPoint(),b.getMultiPoint());
    }
    static Comparator<Tuple2<Trajectory,Double>> ORDER=(a,b)->{
        int c=Double.compare(a._2(),b._2());
        return c!=0?c:Integer.compare(Integer.parseInt(a._1().getId()),Integer.parseInt(b._1().getId()));
    };
    static void regression(Path out)throws Exception{
        Trajectory a=new Trajectory("0",(MultiPoint)WKTUtils.read("MULTIPOINT((-8 41),(-8.001 41),(-8.002 41))"));
        Trajectory b=new Trajectory("1",(MultiPoint)WKTUtils.read("MULTIPOINT((-8 41.002),(-8.001 41.002),(-8.002 41.002))"));
        List<Object> records=new ArrayList<>();
        for(int f=0;f<3;f++){
            CalculateSimilarity orig=new CalculateSimilarity(a.toText(),f);
            CalculateSimilarity copy=(CalculateSimilarity)CalculateSimilarity.parseFrom(orig.toByteArray());
            Cell input=new KeyValue(Bytes.toBytes("r"),Bytes.toBytes(CF),Bytes.toBytes(utils.Constants.GEOM),Bytes.toBytes(b.toText()));
            String wire=Bytes.toString(CellUtil.cloneValue(copy.transformCell(input)));
            String[] parts=wire.split("(?<=\\))-",2);
            if(parts.length!=2||WKTUtils.read(parts[0])==null)throw new AssertionError(wire);
            double actual=Double.parseDouble(parts[1]),expect=distance(a,b,f);
            if(actual!=expect)throw new AssertionError("Metric lost: "+f+" "+actual+" != "+expect);
            records.add(row("func",f,"expected",expect,"serialized_actual",actual));
        }
        String tiny="MULTIPOINT((-8 41))-1E-12";
        if(Double.parseDouble(tiny.split("(?<=\\))-",2)[1])!=1e-12)throw new AssertionError(tiny);
        save(out,row("success",true,"metric_roundtrips",records,"negative_coordinate_parse",true,"negative_exponent_parse",true));
    }
    static void encode(Dataset db,Path out)throws Exception{
        for(int i=0;i<2048;i++)feature(db.get(i));
        List<Object> passes=new ArrayList<>();
        for(int pass=0;pass<3;pass++){
            ExecutorService exec=Executors.newFixedThreadPool(WORKERS);
            AtomicInteger next=new AtomicInteger(),done=new AtomicInteger();
            AtomicLong pivots=new AtomicLong(),polygons=new AtomicLong();
            long start=System.nanoTime(); List<Future<?>> fs=new ArrayList<>();
            for(int w=0;w<WORKERS;w++)fs.add(exec.submit(()->{
                for(int lo;(lo=next.getAndAdd(512))<db.n;){
                    for(int i=lo;i<Math.min(lo+512,db.n);i++){
                        Trajectory t=db.get(i); feature(t);
                        pivots.addAndGet(t.getDPFeature().getIndexes().size());
                        polygons.addAndGet(t.getDPFeature().getMBRs().getNumGeometries());
                    }
                    int n=done.addAndGet(Math.min(512,db.n-lo));
                    if(n%65536==0)System.out.println("ENCODE "+n+"/"+db.n);
                }
            }));
            exec.shutdown();for(Future<?> f:fs)f.get();
            double secs=(System.nanoTime()-start)/1e9;
            passes.add(row("repeat",pass,"seconds",secs,"pivots",pivots.get(),"polygons",polygons.get()));
            save(out,row("success",pass==2,"database",db.n,"workers",WORKERS,"scope","resident float64 coordinates -> Java trajectory objects and all native DP features; no row keys or HBase writes","passes",passes));
        }
    }
    static void ingest(Dataset db,int[] member,String tableName,Path out)throws Exception{
        Configuration conf=HBaseConfiguration.create();
        try(Connection conn=ConnectionFactory.createConnection(conf);Admin admin=conn.getAdmin()){
            TableName table=TableName.valueOf(tableName);
            if(admin.tableExists(table))throw new IOException("Refusing to overwrite existing table "+tableName);
            HTableDescriptor desc=new HTableDescriptor(table);desc.addFamily(new HColumnDescriptor(CF));
            byte[][] splits=new byte[SHARDS-1][];for(int i=1;i<SHARDS;i++)splits[i-1]=new byte[]{(byte)i};
            admin.createTable(desc,splits);
            ExecutorService exec=Executors.newFixedThreadPool(WORKERS);
            AtomicInteger next=new AtomicInteger(),done=new AtomicInteger();
            AtomicLong prepareNs=new AtomicLong(),putNs=new AtomicLong();
            long start=System.nanoTime(); List<Future<?>> futures=new ArrayList<>();
            for(int w=0;w<WORKERS;w++)futures.add(exec.submit(()->{
                PutUtils utils=new PutUtils(G);
                try(BufferedMutator mut=conn.getBufferedMutator(new BufferedMutatorParams(table).writeBufferSize(4*1024*1024))){
                    for(int lo;(lo=next.getAndAdd(128))<member.length;){
                        long t=System.nanoTime();List<Put> puts=new ArrayList<>();
                        for(int i=lo;i<Math.min(lo+128,member.length);i++)puts.add(utils.getPut(db.get(member[i]),SHARDS));
                        prepareNs.addAndGet(System.nanoTime()-t);t=System.nanoTime();
                        mut.mutate(puts);putNs.addAndGet(System.nanoTime()-t);
                        int d=done.addAndGet(puts.size());
                        if(d%16384==0)System.out.println("INGEST "+d+"/"+member.length);
                    }
                    mut.flush();
                }catch(Exception e){throw new RuntimeException(e);}
            }));
            exec.shutdown();for(Future<?> f:futures)f.get();
            double insertSeconds=(System.nanoTime()-start)/1e9;
            long t=System.nanoTime();admin.flush(table);double flushSeconds=(System.nanoTime()-t)/1e9;
            long rows=0;
            try(Table h=conn.getTable(table);ResultScanner scan=h.getScanner(new Scan().setFilter(new FirstKeyOnlyFilter()).setCaching(4096))){
                for(Result r:scan)rows++;
            }
            if(rows!=member.length)throw new AssertionError("Wrong stored count "+rows);
            save(out,row("success",true,"table",tableName,"rows",rows,"workers",WORKERS,"index_ingest_wall_seconds",insertSeconds,
                "flush_request_seconds",flushSeconds,"worker_prepare_cpu_seconds_sum",prepareNs.get()/1e9,"worker_enqueue_seconds_sum",putNs.get()/1e9,
                "note","Index/serialization and HBase insertion overlap across workers; summed worker spans are not wall time. Final row scan verified all inserted rows; flush request is not a standalone durability benchmark."));
        }
    }
    static List<Tuple2<Trajectory,Double>> query(Client client,Trajectory q,int f)throws Exception{
        MinMaxPriorityQueue<Tuple2<Trajectory,Double>> result=client.knnQuery(q,K,INTERVAL,f);
        List<Tuple2<Trajectory,Double>> a=new ArrayList<>(result);a.sort(ORDER);
        Set<String> seen=new HashSet<>();
        for(Tuple2<Trajectory,Double> x:a)if(!seen.add(x._1().getId()))throw new AssertionError("Duplicate result ID");
        if(a.size()!=K)throw new AssertionError("Native query returned "+a.size()+" of "+K);
        return a;
    }
    static void gate(Dataset db,Dataset qs,int[] member,int[] qids,String table,Path out)throws Exception{
        Client client=new Client(G,table,SHARDS);List<Object> records=new ArrayList<>();
        try{
            for(int f:new int[]{0,1,2})for(int qi:qids){
                Trajectory q=qs.get(qi);feature(q);
                List<Tuple2<Trajectory,Double>> brute=new ArrayList<>();
                for(int id:member){Trajectory d=db.get(id);brute.add(new Tuple2<>(d,distance(q,d,f)));}
                brute.sort(ORDER);List<Tuple2<Trajectory,Double>> got=query(client,q,f);
                Set<String> want=new HashSet<>();for(int i=0;i<K;i++)want.add(brute.get(i)._1().getId());
                int hits=0;double worst=brute.get(K-1)._2();
                for(Tuple2<Trajectory,Double> x:got){
                    if(want.contains(x._1().getId()))hits++;
                    if(x._2()>worst+1e-12)throw new AssertionError("Native missed closer trajectory, f="+f+" q="+qi+" got="+x._2()+" kth="+worst);
                    double oracle=distance(q,db.get(Integer.parseInt(x._1().getId())),f);
                    if(Math.abs(x._2()-oracle)>1e-12)throw new AssertionError("Returned wrong distance");
                }
                List<Integer> gotIds=new ArrayList<>();List<Double> gotDistances=new ArrayList<>();
                for(Tuple2<Trajectory,Double> x:got){gotIds.add(Integer.parseInt(x._1().getId()));gotDistances.add(x._2());}
                records.add(row("func",f,"query",qi,"id_hits",hits,"k",K,"distance_exact",true,"ids",gotIds,"distances",gotDistances));
                save(out,row("success",false,"completed",records));
                System.out.println("GATE metric="+f+" query="+qi+" hits="+hits);
            }
        }finally{client.close();}
        save(out,row("success",true,"completed",records));
    }
    static void queryAll(Dataset qs,String table,Path out,int f,int repeat)throws Exception{
        Client client=new Client(G,table,SHARDS);
        int completed=0;
        if(Files.exists(out)){
            for(String line:Files.readAllLines(out)){
                Map prior=new com.fasterxml.jackson.databind.ObjectMapper().readValue(line,Map.class);
                if(((Number)prior.get("query")).intValue()!=completed||((Number)prior.get("func")).intValue()!=f||((Number)prior.get("repeat")).intValue()!=repeat)
                    throw new AssertionError("Invalid query prefix; refuse to overwrite "+out);
                completed++;
            }
        }
        if(completed>qs.n)throw new AssertionError("Too many saved queries");
        System.out.println("RESUME preserved="+completed+" metric="+f+" repeat="+repeat);
        try(PrintWriter writer=new PrintWriter(Files.newBufferedWriter(out,StandardOpenOption.CREATE,StandardOpenOption.APPEND))){
            for(int i=0;i<3;i++){Trajectory q=qs.get(i);feature(q);query(client,q,f);}
            for(int i=completed;i<qs.n;i++){
                long t=System.nanoTime();Trajectory q=qs.get(i);feature(q);double prep=(System.nanoTime()-t)/1e9;
                t=System.nanoTime();List<Tuple2<Trajectory,Double>> got=query(client,q,f);double secs=(System.nanoTime()-t)/1e9;
                List<Integer> resultIds=new ArrayList<>();List<Double> distances=new ArrayList<>();
                for(Tuple2<Trajectory,Double> x:got){resultIds.add(Integer.parseInt(x._1().getId()));distances.add(x._2());}
                Map<String,Object> record=row("query",i,"func",f,"repeat",repeat,"prepare_seconds",prep,"search_seconds",secs,"ids",resultIds,"distances",distances);
                writer.println(new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(record));writer.flush();
                System.out.println("QUERY metric="+f+" repeat="+repeat+" "+(i+1)+"/"+qs.n+" seconds="+secs);
            }
        }finally{client.close();}
    }
    public static void main(String[] args)throws Exception{
        String mode=args[0];Path out=Paths.get(args[2]);
        if(mode.equals("regression")){regression(out);return;}
        try(Dataset db=new Dataset(Paths.get(args[1]),"base");Dataset qs=new Dataset(Paths.get(args[1]),"query")){
            if(mode.equals("encode")){encode(db,out);return;}
            if(mode.equals("ingest")){ingest(db,ids(args[4].equals("all")?null:Paths.get(args[4]),db.n),args[3],out);return;}
            if(mode.equals("gate")){gate(db,qs,ids(Paths.get(args[4]),db.n),ids(Paths.get(args[5]),qs.n),args[3],out);return;}
            if(mode.equals("query")){queryAll(qs,args[3],out,Integer.parseInt(args[4]),Integer.parseInt(args[5]));return;}
            throw new IllegalArgumentException(mode);
        }
    }
}
